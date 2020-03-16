mod utils;

use wasm_bindgen::prelude::*;
use js_sys;

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

pub trait Array {
    fn get();
}

#[wasm_bindgen]
pub struct Ndarray {
    data : Data,
    shape : Vec<usize>,
    strides : Vec<usize>,
    ndim : usize,
}

#[wasm_bindgen]
impl Ndarray {
    #[wasm_bindgen(constructor)]
    pub fn new(input: js_sys::Array) -> Ndarray{
        let mut shape : Vec<usize>= Vec::new();
        let array = Ndarray::flatten_jsarray(input.to_vec(),&mut shape);
        Ndarray{
            data: Data::F64(Rc::from(array)),
            ndim : shape.len(),
            shape : shape.clone(),
            strides : shape,
        }
    }
}

impl Ndarray {
    fn flatten_jsarray(input: Vec<JsValue>, shape : &mut Vec<usize>) -> Vec<f64> {
        match js_sys::Array::is_array(&input[0]) {
            true => {
                shape.push(input.len());
                let array = input.into_iter().flat_map(|x| js_sys::Array::from(&x).to_vec()).collect();
                Ndarray::flatten_jsarray(array, shape)
            }
            false => {
                let array : Vec<f64> = input.into_iter().map(|x : JsValue| x.as_f64().unwrap()).collect();
                shape.push(array.len());
                array
            },
        }
    }
}

impl Array for Ndarray {
    fn get() {
        unimplemented!()
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

impl Array for NdarrayView {
    fn get() {
        unimplemented!()
    }
}

#[wasm_bindgen]
pub struct NdarrayViewMut {
    data : DataViewMut
}

#[wasm_bindgen]
impl NdarrayViewMut {
    fn new(ndarray: &mut Ndarray) -> Result<NdarrayViewMut,JsValue> {

        match ndarray.data {
            Data::I32(ref mut data) =>
            match Rc::get_mut(data) {
                Some(mut_ref) => Ok(NdarrayViewMut{
                    data : DataViewMut::I32(mut_ref),
                }),
                None => Err(JsValue::from_str("Data must have single owner to be mutated"))
            },
            Data::F64(ref mut data) =>
            match Rc::get_mut(data) {
                Some(mut_ref) => Ok(NdarrayViewMut{
                    data : DataViewMut::F64(mut_ref),
                }),
                None => Err(JsValue::from_str("Data must have single owner to be mutated"))
            }
        }

    }
}
