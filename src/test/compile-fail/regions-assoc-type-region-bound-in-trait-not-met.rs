// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the compiler checks that arbitrary region bounds declared
// in the trait must be satisfied on the impl. Issue #20890.

trait Foo<'a> {
    type Value: 'a;
    fn dummy(&'a self) { }
}

impl<'a> Foo<'a> for &'a i16 {
    // OK.
    type Value = &'a i32;
}

impl<'a> Foo<'static> for &'a i32 {
    //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting
    //~| ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting
    //~| ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting
    //~| NOTE cannot infer an appropriate lifetime
    //~| NOTE first, the lifetime cannot outlive the lifetime 'a as defined on the impl
    //~| NOTE ...so that trait type parameters matches those specified on the impl
    //~| NOTE ...so that the type `&i32` will meet its required lifetime bounds
    //~| NOTE but, the lifetime must be valid for the static lifetime
    //~| NOTE but, the lifetime must be valid for the static lifetime
    //~| NOTE but, the lifetime must be valid for the static lifetime
    type Value = &'a i32;
}

impl<'a,'b> Foo<'b> for &'a i64 {
    //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting
    //~| ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting
    //~| ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting
    //~| NOTE cannot infer an appropriate lifetime
    //~| NOTE first, the lifetime cannot outlive the lifetime 'a as defined on the impl
    //~| NOTE ...so that trait type parameters matches those specified on the impl
    //~| NOTE but, the lifetime must be valid for the lifetime 'b as defined on the impl
    //~| NOTE ...so that the type `&i32` will meet its required lifetime bounds
    type Value = &'a i32;
}

fn main() { }
