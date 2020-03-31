//! Efficient wasm-implementation of an n-dimensional Array for Javascript
//!
//! Wasm restrictions on Rust (at least at wasm boundary):
//! - No generics
//! - No polymorphims (no traits)
//! - No lifetimes
//!
//! Solutions:
//! - no generics => enum NdarrayUnion for different type parameters for Ndarray
//! - no polymorphism => enum Subview for different behavior of ndarrays
//! - no lifetimes => Using reference counting (std::rc::Rc), unsafe
//!
//! Using enums requires minimally more memory (2 enums = 2 Byte)

mod iter;
mod js_interop;
pub mod ndarray;
mod utils;

use wasm_bindgen::prelude::*;

pub use ndarray::*;

// #[wasm_bindgen(start)]
// pub fn main() {
//     utils::set_panic_hook();
// }
