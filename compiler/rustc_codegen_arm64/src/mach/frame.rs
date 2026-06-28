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
pub(crate) fn align_up(value: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// The resolved stack-frame layout for one function.
///
/// Constructed with the outgoing-argument area size (determined by a pre-pass over the function's
/// calls); local spill slots are then allocated against it during instruction selection.
#[derive(Debug)]
pub struct FrameLayout {
    /// Size of the outgoing-argument area at the bottom of the frame (`sp + 0 .. sp + outgoing`),
    /// rounded up to the 16-byte stack alignment.
    outgoing_size: u32,
    /// `sp`-relative offset at which the next local slot will be placed; starts just above the
    /// outgoing area and grows as slots are allocated.
    next_local: u32,
}

impl FrameLayout {
    /// Reserve `outgoing_bytes` (rounded up to the 16-byte stack alignment) for outgoing call
    /// arguments at the bottom of the frame.
    pub fn new(outgoing_bytes: u32) -> FrameLayout {
        let outgoing_size = align_up(outgoing_bytes, 16);
        FrameLayout { outgoing_size, next_local: outgoing_size }
    }

    /// Allocate a local spill slot of `size`/`align` bytes, returning its `sp`-relative offset.
    pub fn alloc_local(&mut self, size: u64, align: u64) -> u32 {
        let align = (align.max(1)) as u32;
        let off = align_up(self.next_local, align);
        self.next_local = off + (size.max(1)) as u32;
        off
    }

    /// The `sp`-relative offset of the outgoing-argument slot at byte position `arg_offset`.
    ///
    /// Outgoing arguments occupy `sp + 0 .. sp + outgoing_size`; callers must keep `arg_offset`
    /// within that range (guaranteed by the outgoing-area pre-pass).
    pub fn outgoing_arg(&self, arg_offset: u32) -> u32 {
        debug_assert!(arg_offset < self.outgoing_size || self.outgoing_size == 0);
        arg_offset
    }

    /// The `fp`-relative offset of an incoming stack argument at byte position `arg_offset`.
    ///
    /// Our caller places stack arguments immediately above our saved `fp`/`lr`, i.e. at
    /// `fp + 16 + arg_offset`.
    pub fn incoming_arg(&self, arg_offset: u32) -> u32 {
        16 + arg_offset
    }

    /// The bytes reserved for the outgoing-argument area.
    pub fn outgoing_size(&self) -> u32 {
        self.outgoing_size
    }

    /// The total frame size (outgoing area + local slots), rounded up to the 16-byte stack
    /// alignment. This is what the prologue subtracts from `sp`.
    pub fn frame_size(&self) -> u32 {
        align_up(self.next_local, 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frame() {
        let frame = FrameLayout::new(0);
        assert_eq!(frame.outgoing_size(), 0);
        assert_eq!(frame.frame_size(), 0);
    }

    #[test]
    fn outgoing_is_16_aligned() {
        assert_eq!(FrameLayout::new(0).outgoing_size(), 0);
        assert_eq!(FrameLayout::new(8).outgoing_size(), 16);
        assert_eq!(FrameLayout::new(16).outgoing_size(), 16);
        assert_eq!(FrameLayout::new(24).outgoing_size(), 32);
        assert_eq!(FrameLayout::new(120).outgoing_size(), 128);
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
    fn outgoing_args_start_at_zero() {
        let frame = FrameLayout::new(32);
        assert_eq!(frame.outgoing_arg(0), 0);
        assert_eq!(frame.outgoing_arg(8), 8);
    }
}
