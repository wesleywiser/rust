fn main() {
    let x: &[u32] = &[1, 2, 3];
    x[1];
}

// END RUST SOURCE
// START rustc.main.ConstProp.before.mir
//  bb0: {
//      ...
//      _3 = &(promoted[0]: [u32; 3]);
//      _2 = _3;
//      _1 = move _2 as &[u32] (Pointer(Unsize));
//      ...
//      _6 = const 1usize;
//      _7 = Len((*_1));
//      _8 = Lt(_6, _7);
//      assert(move _8, "index out of bounds: the len is move _7 but the index is _6") -> bb1;
//  }
//  bb1: {
//      _5 = (*_1)[_6];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.before.mir
// START rustc.main.ConstProp.after.mir
//  bb0: {
//      ...
//      _3 = &(promoted[0]: [u32; 3]);
//      _2 = _3;
//      _1 = move _2 as &[u32] (Pointer(Unsize));
//      ...
//      _6 = const 1usize;
//      _7 = Len((*_1));
//      _8 = Lt(_6, _7);
//      assert(move _8, "index out of bounds: the len is move _7 but the index is _6") -> bb1;
//  }
//  bb1: {
//      _5 = (*_1)[_6];
//      ...
//      return;
//  }
// END rustc.main.ConstProp.after.mir
