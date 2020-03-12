mod utils;

use wasm_bindgen::prelude::*;

use std::rc::Rc;
use std::rc::Weak;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

enum Data {
    I32(Vec<i32>),
    F64(Vec<f64>),
}

#[wasm_bindgen]
pub struct Ndarray {
    data : Rc<Data>,
}

#[wasm_bindgen]
impl Ndarray {
    #[wasm_bindgen(constructor)]
    pub fn new_i32(input: Box<[i32]>) -> Ndarray {
        Ndarray{
            data : Rc::new(Data::I32(Vec::from(input))),
        }
    }
    #[wasm_bindgen(constructor)]
    pub fn new_f64(input: Box<[f64]>) -> Ndarray {
        Ndarray{
            data : Rc::new(Data::F64(Vec::from(input))),
        }
    }
}

#[wasm_bindgen]
pub struct NdarrayView {
    data : Weak<Data>
}

#[wasm_bindgen]
impl NdarrayView {
    fn new(ndarray: &Ndarray) -> NdarrayView {
        NdarrayView {
            data : Rc::downgrade(&ndarray.data),
        }

    }
}
