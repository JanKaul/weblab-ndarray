use std::rc::Rc;
use wasm_bindgen::prelude::*;

use super::ndarray::{Data, Ndarray};

enum DataMut {
    I32(*mut [i32]),
    F64(*mut [f64]),
}

#[wasm_bindgen]
pub struct NdarrayMut {
    data: DataMut,
}

#[wasm_bindgen]
impl NdarrayMut {
    pub fn new(ndarray: &mut Ndarray) -> Result<NdarrayMut, JsValue> {
        match ndarray.data() {
            Data::I32(ref mut data) => match Rc::get_mut(data) {
                Some(mut_ref) => Ok(NdarrayMut {
                    data: DataMut::I32(mut_ref),
                }),
                None => Err(JsValue::from_str(
                    "Data must have single owner to be mutated",
                )),
            },
            Data::F64(ref mut data) => match Rc::get_mut(data) {
                Some(mut_ref) => Ok(NdarrayMut {
                    data: DataMut::F64(mut_ref),
                }),
                None => Err(JsValue::from_str(
                    "Data must have single owner to be mutated",
                )),
            },
        }
    }
}
