use rustc_middle::mir::mono::{CodegenUnit, InstantiationMode, Visibility, MonoItem, Linkage};
use super::{default::mono_item_linkage_and_visibility, Partitioner, merging, MonoItemPlacement};
use rustc_middle::ty::TyCtxt;
use rustc_data_structures::{fx::FxHashMap, stable_set::FxHashSet};
use rustc_span::Symbol;
use crate::monomorphize::collector::InliningMap;
use std::collections::hash_map::Entry;

pub struct DebugPartioning;

const TARGET_CGU_SIZE: usize = 2000;

impl<'tcx> Partitioner<'tcx> for DebugPartioning {
    fn place_root_mono_items(
        &mut self,
        tcx: TyCtxt<'tcx>,
        mono_items: &mut dyn Iterator<Item = MonoItem<'tcx>>,
    ) -> super::PreInliningPartitioning<'tcx> {
        let mut roots = FxHashSet::default();
        let mut codegen_units = Vec::new();
        let mut internalization_candidates = FxHashSet::default();

        let mut estimated_size = 0;
        codegen_units.push(CodegenUnit::new(Symbol::intern("cgu_0")));

        for mono_item in mono_items {
            match mono_item.instantiation_mode(tcx) {
                InstantiationMode::GloballyShared { .. } => {},
                InstantiationMode::LocalCopy => continue,
            }

            if estimated_size > TARGET_CGU_SIZE {
                codegen_units.push(CodegenUnit::new(Symbol::intern(&format!("cgu_{}", codegen_units.len()))));
                estimated_size = 0;
            }

            estimated_size += mono_item.size_estimate(tcx);

            let mut can_be_internalized = true;
            let (linkage, visibility) = mono_item_linkage_and_visibility(tcx, &mono_item, &mut can_be_internalized, false);
            if visibility == Visibility::Hidden && can_be_internalized {
                internalization_candidates.insert(mono_item);
            }

            let cgu = codegen_units.last_mut().expect("there must be at least one cgu");
            cgu.items_mut().insert(mono_item, (linkage, visibility));
            roots.insert(mono_item);
        }

        super::PreInliningPartitioning {
            codegen_units,
            roots,
            internalization_candidates,
        }
    }

    fn merge_codegen_units(
        &mut self,
        tcx: TyCtxt<'tcx>,
        initial_partitioning: &mut super::PreInliningPartitioning<'tcx>,
        target_cgu_count: usize,
    ) {
        // TODO: split big CGUs?
        merging::merge_codegen_units(tcx, initial_partitioning, target_cgu_count);
    }

    fn place_inlined_mono_items(
        &mut self,
        initial_partitioning: super::PreInliningPartitioning<'tcx>,
        inlining_map: &crate::monomorphize::collector::InliningMap<'tcx>,
    ) -> super::PostInliningPartitioning<'tcx> {
        let mut new_partitioning = Vec::new();
        let mut mono_item_placements = FxHashMap::default();

        let super::PreInliningPartitioning {
            codegen_units: initial_cgus,
            roots,
            internalization_candidates,
        } = initial_partitioning;

        let single_codegen_unit = initial_cgus.len() == 1;

        for old_codegen_unit in initial_cgus {
            // Collect all items that need to be available in this codegen unit.
            let mut reachable = FxHashSet::default();
            for root in old_codegen_unit.items().keys() {
                follow_inlining(*root, inlining_map, &mut reachable);
            }

            let mut new_codegen_unit = CodegenUnit::new(old_codegen_unit.name());

            // Add all monomorphizations that are not already there.
            for mono_item in reachable {
                if let Some(linkage) = old_codegen_unit.items().get(&mono_item) {
                    // This is a root, just copy it over.
                    new_codegen_unit.items_mut().insert(mono_item, *linkage);
                } else {
                    if roots.contains(&mono_item) {
                        bug!(
                            "GloballyShared mono-item inlined into other CGU: \
                              {:?}",
                            mono_item
                        );
                    }

                    // This is a CGU-private copy.
                    new_codegen_unit
                        .items_mut()
                        .insert(mono_item, (Linkage::Internal, Visibility::Default));
                }

                if !single_codegen_unit {
                    // If there is more than one codegen unit, we need to keep track
                    // in which codegen units each monomorphization is placed.
                    match mono_item_placements.entry(mono_item) {
                        Entry::Occupied(e) => {
                            let placement = e.into_mut();
                            debug_assert!(match *placement {
                                MonoItemPlacement::SingleCgu { cgu_name } => {
                                    cgu_name != new_codegen_unit.name()
                                }
                                MonoItemPlacement::MultipleCgus => true,
                            });
                            *placement = MonoItemPlacement::MultipleCgus;
                        }
                        Entry::Vacant(e) => {
                            e.insert(MonoItemPlacement::SingleCgu {
                                cgu_name: new_codegen_unit.name(),
                            });
                        }
                    }
                }
            }

            new_partitioning.push(new_codegen_unit);
        }

        return super::PostInliningPartitioning {
            codegen_units: new_partitioning,
            mono_item_placements,
            internalization_candidates,
        };

        fn follow_inlining<'tcx>(
            mono_item: MonoItem<'tcx>,
            inlining_map: &InliningMap<'tcx>,
            visited: &mut FxHashSet<MonoItem<'tcx>>,
        ) {
            if !visited.insert(mono_item) {
                return;
            }

            inlining_map.with_inlining_candidates(mono_item, |target| {
                follow_inlining(target, inlining_map, visited);
            });
        }
    }

    fn internalize_symbols(
        &mut self,
        _tcx: TyCtxt<'tcx>,
        partitioning: &mut super::PostInliningPartitioning<'tcx>,
        inlining_map: &crate::monomorphize::collector::InliningMap<'tcx>,
    ) {
        // TODO: It might be good to track how many duplicated symbols we internalize
        // since that means we have the same code in multiple cgus which hurts compile time.

        if partitioning.codegen_units.len() == 1 {
            // Fast path for when there is only one codegen unit. In this case we
            // can internalize all candidates, since there is nowhere else they
            // could be accessed from.
            for cgu in &mut partitioning.codegen_units {
                for candidate in &partitioning.internalization_candidates {
                    cgu.items_mut().insert(*candidate, (Linkage::Internal, Visibility::Default));
                }
            }

            return;
        }

        // Build a map from every monomorphization to all the monomorphizations that
        // reference it.
        let mut accessor_map: FxHashMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>> = Default::default();
        inlining_map.iter_accesses(|accessor, accessees| {
            for accessee in accessees {
                accessor_map.entry(*accessee).or_default().push(accessor);
            }
        });

        let mono_item_placements = &partitioning.mono_item_placements;

        // For each internalization candidates in each codegen unit, check if it is
        // accessed from outside its defining codegen unit.
        for cgu in &mut partitioning.codegen_units {
            let home_cgu = MonoItemPlacement::SingleCgu { cgu_name: cgu.name() };

            for (accessee, linkage_and_visibility) in cgu.items_mut() {
                if !partitioning.internalization_candidates.contains(accessee) {
                    // This item is no candidate for internalizing, so skip it.
                    continue;
                }
                debug_assert_eq!(mono_item_placements[accessee], home_cgu);

                if let Some(accessors) = accessor_map.get(accessee) {
                    if accessors
                        .iter()
                        .filter_map(|accessor| {
                            // Some accessors might not have been
                            // instantiated. We can safely ignore those.
                            mono_item_placements.get(accessor)
                        })
                        .any(|placement| *placement != home_cgu)
                    {
                        // Found an accessor from another CGU, so skip to the next
                        // item without marking this one as internal.
                        continue;
                    }
                }

                // If we got here, we did not find any accesses from other CGUs,
                // so it's fine to make this monomorphization internal.
                *linkage_and_visibility = (Linkage::Internal, Visibility::Default);
            }
        }
    }
}