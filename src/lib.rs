//! # Weblab-ndarray

mod ndarray;
mod ndarray_mut;
mod utils;

use wasm_bindgen::prelude::*;

use std::rc::Rc;

#[wasm_bindgen(start)]
pub fn main() {
    utils::set_panic_hook();
}

#[wasm_bindgen]
pub struct TestArray {
    data: Rc<[f64]>,
}

#[wasm_bindgen]
impl TestArray {
    #[wasm_bindgen(constructor)]
    pub fn new(input: &js_sys::Float64Array) -> TestArray {
        TestArray {
            data: Rc::from(input.to_vec()),
        }
    }
    pub fn mul(&self, other: &TestArray) -> TestArray {
        // let result : Rc<[f64]> = self.data.into_par_iter().zip(other.data.into_par_iter()).map(|(x,y)| x*y).collect::<Vec<f64>>().into();
        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x * y)
            .collect::<Rc<[f64]>>();
        TestArray { data: result }
    }
}
