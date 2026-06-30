//! Stack-frame layout.
//!
//! The layout of a function's frame is decided up front, before any instructions are selected, so
//! that every stack reference can be emitted with its final `sp`/`fp`-relative offset in a single
//! pass (no later rewriting of the instruction stream). The only piece that must be known ahead of
//! time is the size of the *outgoing argument area* — the bytes a call needs to pass arguments that
//! do not fit in registers — which is computed by scanning the function's calls (see
//! `builder::outgoing_arg_bytes`).
//!
//! Frame regions, from low to high address (`sp` sits at the bottom after the prologue):
//!
//! ```text
//!   sp + 0                    outgoing call arguments (passed on the stack to callees)
//!   sp + outgoing_size        local spill slots (grow upward as they are allocated)
//!   sp + frame_size   = fp    saved frame pointer / link register (written by the prologue)
//!   fp + 16                   incoming stack arguments (provided to us by our caller)
//! ```
//!
//! Because the outgoing area is reserved at the bottom, a local slot's offset is simply its
//! position within the locals region added to `outgoing_size`; this is baked in at allocation time
//! rather than fixed up afterwards.

/// Align `value` up to the next multiple of `align` (which must be a power of two).
pub(crate) fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// The resolved stack-frame layout for one function.
///
/// Constructed with the outgoing-argument area size (determined by a pre-pass over the function's
/// calls); local spill slots are then allocated against it during instruction selection.
#[derive(Debug)]
pub struct FrameLayout {
    /// `sp`-relative offset at which the next local slot will be placed; starts just above the
    /// outgoing area and grows as slots are allocated.
    next_local: u64,
    /// The largest alignment requested by any local slot. The stack pointer is only guaranteed
    /// 16-byte aligned on entry, so if any slot needs more the prologue must dynamically realign
    /// `sp` (see `builder::FunctionBuild::finish`).
    max_align: u64,
}

impl FrameLayout {
    /// Reserve `outgoing_bytes` (rounded up to the 16-byte stack alignment) for outgoing call
    /// arguments at the bottom of the frame.
    pub fn new(outgoing_bytes: u64) -> FrameLayout {
        FrameLayout { next_local: align_up(outgoing_bytes, 16), max_align: 16 }
    }

    /// Allocate a local spill slot of `size`/`align` bytes, returning its `sp`-relative offset.
    pub fn alloc_local(&mut self, size: u64, align: u64) -> u64 {
        let align = align.max(1);
        self.max_align = self.max_align.max(align);
        let off = align_up(self.next_local, align);
        self.next_local = off + size.max(1);
        off
    }

    /// The alignment the frame base (`sp`) must satisfy: the maximum over all local slots, never
    /// below the 16-byte AArch64 stack alignment. When this exceeds 16 the prologue realigns `sp`.
    pub fn alignment(&self) -> u64 {
        self.max_align
    }

    /// The `fp`-relative offset of an incoming stack argument at byte position `arg_offset`.
    ///
    /// Our caller places stack arguments immediately above our saved `fp`/`lr`, i.e. at
    /// `fp + 16 + arg_offset`.
    pub fn incoming_arg(&self, arg_offset: u64) -> u64 {
        16 + arg_offset
    }

    /// The total frame size (outgoing area + local slots), rounded up to the 16-byte stack
    /// alignment. This is what the prologue subtracts from `sp`.
    pub fn frame_size(&self) -> u64 {
        align_up(self.next_local, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frame() {
        let frame = FrameLayout::new(0);
        assert_eq!(frame.frame_size(), 0);
    }

    #[test]
    fn outgoing_is_16_aligned() {
        // With no locals allocated, the frame is exactly the (16-byte-rounded) outgoing area.
        assert_eq!(FrameLayout::new(0).frame_size(), 0);
        assert_eq!(FrameLayout::new(8).frame_size(), 16);
        assert_eq!(FrameLayout::new(16).frame_size(), 16);
        assert_eq!(FrameLayout::new(24).frame_size(), 32);
        assert_eq!(FrameLayout::new(120).frame_size(), 128);
    }

    #[test]
    fn locals_start_above_outgoing_area() {
        // With a 24-byte outgoing area (rounded to 32), the first local sits at offset 32.
        let mut frame = FrameLayout::new(24);
        assert_eq!(frame.alloc_local(8, 8), 32);
        assert_eq!(frame.alloc_local(8, 8), 40);
        // 48 bytes used -> frame rounds up to 48 (already 16-aligned).
        assert_eq!(frame.frame_size(), 48);
    }

    #[test]
    fn locals_are_aligned() {
        let mut frame = FrameLayout::new(0);
        assert_eq!(frame.alloc_local(1, 1), 0); // a byte
        // next slot needs 8-byte alignment -> padded from 1 up to 8.
        assert_eq!(frame.alloc_local(8, 8), 8);
        // a 4-byte slot at the next 4-aligned offset (16).
        assert_eq!(frame.alloc_local(4, 4), 16);
        assert_eq!(frame.frame_size(), 32); // 20 bytes used, rounded to 16.
    }

    #[test]
    fn frame_size_is_16_aligned() {
        let mut frame = FrameLayout::new(0);
        frame.alloc_local(8, 8); // 8 bytes
        assert_eq!(frame.frame_size(), 16);
    }

    #[test]
    fn incoming_args_are_above_saved_registers() {
        let frame = FrameLayout::new(0);
        assert_eq!(frame.incoming_arg(0), 16);
        assert_eq!(frame.incoming_arg(8), 24);
    }

    #[test]
    fn tracks_max_alignment() {
        // No locals, or only normally-aligned ones: the base needs only the 16-byte stack alignment.
        let mut frame = FrameLayout::new(0);
        assert_eq!(frame.alignment(), 16);
        frame.alloc_local(8, 8);
        assert_eq!(frame.alignment(), 16);
        // An over-aligned local bumps the required base alignment (triggers prologue realignment).
        frame.alloc_local(32, 32);
        assert_eq!(frame.alignment(), 32);
        // A larger one wins; a smaller later one does not lower it.
        frame.alloc_local(64, 64);
        frame.alloc_local(8, 8);
        assert_eq!(frame.alignment(), 64);
        // The over-aligned slot's own offset is aligned within the frame, too.
        let off = frame.alloc_local(16, 128);
        assert_eq!(off % 128, 0);
        assert_eq!(frame.alignment(), 128);
    }
}
