//! # Weblab-ndarray

mod iter;
mod js_interop;
mod ndarray;
mod utils;

use wasm_bindgen::prelude::*;

use std::rc::Rc;

pub use ndarray::*;

#[wasm_bindgen(start)]
pub fn main() {
    utils::set_panic_hook();
}

enum TestData {
    F64(Rc<[f64]>),
}

#[wasm_bindgen]
pub struct TestArray {
    data: TestData,
}

#[wasm_bindgen]
impl TestArray {
    #[wasm_bindgen(constructor)]
    pub fn new(input: &js_sys::Float64Array) -> TestArray {
        TestArray {
            data: TestData::F64(Rc::from(input.to_vec())),
        }
    }
    pub fn mul(&self, other: &TestArray) -> TestArray {
        // let result : Rc<[f64]> = self.data.into_par_iter().zip(other.data.into_par_iter()).map(|(x,y)| x*y).collect::<Vec<f64>>().into();
        match (&self.data, &other.data) {
            (TestData::F64(one), TestData::F64(two)) => {
                let result = one
                    .iter()
                    .zip(two.iter())
                    .map(|(x, y)| x * y)
                    .collect::<Rc<[f64]>>();
                TestArray {
                    data: TestData::F64(result),
                }
            }
        }
    }
}
