#![feature(rustc_attrs)]

enum Abc {
    A(u8),
    B(i8),
    C,
    D,
}

#[rustc_mir(graphviz = "output.dot")]
fn foo(x: Abc) -> i32 {
    match x {
        Abc::C => 3,
        Abc::D => 4,
        Abc::B(_) => 2,
        Abc::A(_) => 1,
    }
}

fn main() {
    assert_eq!(4, foo(Abc::D));
}
