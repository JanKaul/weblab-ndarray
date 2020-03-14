mod utils;

use wasm_bindgen::prelude::*;

use std::rc::Rc;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

enum Data {
    I32(Rc<[i32]>),
    F64(Rc<[f64]>),
}

enum DataViewMut {
    I32(*mut [i32]),
    F64(*mut [f64]),
}

#[wasm_bindgen]
pub struct Ndarray {
    data : Data,
}

#[wasm_bindgen]
impl Ndarray {
    #[wasm_bindgen(constructor)]
    pub fn new_i32(input: Box<[i32]>) -> Ndarray {
        Ndarray{
            data : Data::I32(Rc::from(input)),
        }
    }
    #[wasm_bindgen(constructor)]
    pub fn new_f64(input: Box<[f64]>) -> Ndarray {
        Ndarray{
            data : Data::F64(Rc::from(input)),
        }
    }
}

#[wasm_bindgen]
pub struct NdarrayView {
    data : Data
}

#[wasm_bindgen]
impl NdarrayView {
    fn new(ndarray: &Ndarray) -> NdarrayView {
        match &ndarray.data {
            Data::I32(data) => NdarrayView{
                data : Data::I32(data.clone()),
            },
            Data::F64(data) => NdarrayView{
                data : Data::F64(data.clone()),
            }
        }

    }
}

#[wasm_bindgen]
pub struct NdarrayViewMut {
    data : DataViewMut
}

#[wasm_bindgen]
impl NdarrayViewMut {
    fn new(ndarray: &mut Ndarray) -> NdarrayViewMut {
        match ndarray.data {
            Data::I32(ref mut data1) => NdarrayViewMut{
                data : DataViewMut::I32(Rc::get_mut(data1).unwrap()),
            },
            Data::F64(ref mut data1) => NdarrayViewMut{
                data : DataViewMut::F64(Rc::get_mut(data1).unwrap()),
            }
        }

    }
}
