mod utils;

use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, weblab!");
}

enum Data {
    I64(&mut [i64]),
    F64(&mut [f64]),
}

#[wasm_bindgen]
pub struct Ndarray {
    data : Data
}

#[wasm_bindgen]
impl Ndarray {
    pub fn new(input: &mut [i64]) -> Ndarray {
        let data = input;

        Ndarray{
            data
        }
    }
    pub fn new(input: &mut [f64]) -> Ndarray {
        let data = input;

        Ndarray{
            data
        }
    }
}
