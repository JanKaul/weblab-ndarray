# weblab-ndarray

Efficient wasm-implementation of an n-dimensional Array for Javascript

Wasm restrictions on Rust (at least at wasm boundary):
- No generics
- No polymorphims (no traits)
- No lifetimes

Solutions:
- no generics => enum NdarrayUnion for different type parameters for Ndarray
- no polymorphism => enum Subview for different behavior of ndarrays
- no lifetimes => Using if reference counting (std::rc::Rc), unsafe

Using enums requires minimally more memory (2 enums = 2 Byte)
