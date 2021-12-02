// compile-flags: -Zinstrument-coverage -Ccodegen-units=4 --crate-type dylib -Copt-level=0
// build-pass

#![allow(dead_code)]

mod foo {
    #[inline(always)]
    pub fn called() { }

    fn uncalled() { }
}

pub mod bar {
    pub fn call_me() {
        super::foo::called();
    }
}

pub mod baz {
    pub fn call_me() {
        super::foo::called();
    }
}
