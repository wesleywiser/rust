use crate::abi::call::{ArgAbi, FnAbi, Reg, RegKind};
use crate::abi::{Abi, Size};

// Win64 ABI: https://docs.microsoft.com/en-us/cpp/build/parameter-passing

pub fn compute_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>) {
    let fixup = |a: &mut ArgAbi<'_, Ty>, is_ret: bool| {
        match a.layout.abi {
            Abi::Uninhabited | Abi::Aggregate { sized: false } => {}
            Abi::ScalarPair(..) | Abi::Aggregate { sized: true } => match a.layout.size.bits() {
                8 => a.cast_to(Reg::i8()),
                16 => a.cast_to(Reg::i16()),
                32 => a.cast_to(Reg::i32()),
                64 => a.cast_to(Reg::i64()),
                _ => a.make_indirect(),
            },
            Abi::Vector { .. } => {
                // FIXME(eddyb) there should be a size cap here
                // (probably what clang calls "illegal vectors").
            }
            Abi::Scalar(_) => {
                if a.layout.size.bytes() > 8 {
                    if is_ret {
                        a.cast_to(Reg {
                            kind: RegKind::Vector,
                            size: Size::from_bytes(a.layout.size.bytes()),
                        });
                    } else {
                        a.make_indirect();
                    }
                } else {
                    a.extend_integer_width_to(32);
                }
            }
        }
    };

    if !fn_abi.ret.is_ignore() {
        fixup(&mut fn_abi.ret, true);
    }
    for arg in fn_abi.args.iter_mut() {
        if arg.is_ignore() {
            continue;
        }
        fixup(arg, false);
    }
}
