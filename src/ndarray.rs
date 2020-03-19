use std::rc::Rc;

use js_sys;
use wasm_bindgen::prelude::*;

use crate::js_interop;

#[wasm_bindgen]
pub struct Ndarray(NdarrayUnion);

pub enum NdarrayUnion {
    I32(NdarrayBase<i32>),
    F64(NdarrayBase<f64>),
}

pub struct NdarrayBase<T> {
    data: Rc<[T]>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> NdarrayBase<T> {
    pub fn data(&self) -> &Rc<[T]> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Rc<[T]> {
        &mut self.data
    }
}

#[wasm_bindgen]
impl Ndarray {
    #[wasm_bindgen(constructor)]
    pub fn new(input: JsValue) -> Result<Ndarray, JsValue> {
        match js_interop::unwrap_js_value(input)? {
            js_interop::JsType::Array(array) => {
                let mut shape: Vec<usize> = Vec::new();
                let flat_array = js_interop::flatten_jsarray(array, &mut shape);
                let data: Rc<[f64]> = flat_array.iter().map(|x| x.as_f64().unwrap()).collect();
                Ok(Ndarray(NdarrayUnion::F64(NdarrayBase {
                    data: data,
                    strides: Ndarray::get_strides_from_shape(&shape),
                    shape: shape,
                })))
            }
            js_interop::JsType::Int32Array(array) => {
                let vec = array.to_vec();
                Ok(Ndarray(NdarrayUnion::I32(NdarrayBase {
                    shape: vec![vec.len()],
                    data: Rc::from(vec),
                    strides: vec![1],
                })))
            }
            js_interop::JsType::Float64Array(array) => {
                let vec = array.to_vec();
                Ok(Ndarray(NdarrayUnion::F64(NdarrayBase {
                    shape: vec![vec.len()],
                    data: Rc::from(vec),
                    strides: vec![1],
                })))
            }
            _ => Err(JsValue::from_str("Input must be some kind of Array")),
        }
    }

    pub fn reshape(&mut self, shape: &js_sys::Array) -> Result<(), JsValue> {
        let vec = js_interop::into_vec_usize(shape)?;
        if vec.iter().product::<usize>() == self.shape().iter().product::<usize>() {
            self.set_strides(Ndarray::get_strides_from_shape(&vec));
            Ok(())
        } else {
            Err(JsValue::from_str("Shape doesn't fit data"))
        }
    }

    pub fn get(&self, indices: Vec<usize>) -> JsValue {
        // TODO: introduce bound checks
        assert_eq!(self.strides().len(), indices.len());
        let index = self
            .strides()
            .iter()
            .zip(indices.iter())
            .map(|(x, y)| x * y)
            .sum::<usize>();
        match &self.0 {
            NdarrayUnion::I32(ndarray) => JsValue::from_f64(ndarray.data[index] as f64),
            NdarrayUnion::F64(ndarray) => JsValue::from_f64(ndarray.data[index]),
        }
    }

    pub fn slice(input: &Ndarray) -> Ndarray {
        match &input.0 {
            NdarrayUnion::I32(ndarray) => Ndarray(NdarrayUnion::I32(NdarrayBase {
                data: ndarray.data.clone(),
                strides: ndarray.strides.clone(),
                shape: ndarray.shape.clone(),
            })),
            NdarrayUnion::F64(ndarray) => Ndarray(NdarrayUnion::F64(NdarrayBase {
                data: ndarray.data.clone(),
                strides: ndarray.strides.clone(),
                shape: ndarray.shape.clone(),
            })),
        }
    }
}

impl Ndarray {
    fn get_strides_from_shape(shape: &Vec<usize>) -> Vec<usize> {
        let mut m = 1;
        shape
            .iter()
            .rev()
            .map(|x| {
                let n = m;
                m = m * x;
                n
            })
            .collect::<Vec<usize>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn strides(&self) -> &Vec<usize> {
        match &self.0 {
            NdarrayUnion::I32(ndarray) => &ndarray.strides,
            NdarrayUnion::F64(ndarray) => &ndarray.strides,
        }
    }

    pub fn set_strides(&mut self, strides: Vec<usize>) {
        match &mut self.0 {
            NdarrayUnion::I32(ndarray) => ndarray.strides = strides,
            NdarrayUnion::F64(ndarray) => ndarray.strides = strides,
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        match &self.0 {
            NdarrayUnion::I32(ndarray) => &ndarray.shape,
            NdarrayUnion::F64(ndarray) => &ndarray.shape,
        }
    }
}

#[wasm_bindgen]
pub struct NdarrayMut(NdarrayUnionMut);

#[derive(Debug)]
pub struct NdarrayBaseMut<T> {
    data: *mut [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Debug)]
pub enum NdarrayUnionMut {
    I32(NdarrayBaseMut<i32>),
    F64(NdarrayBaseMut<f64>),
}

#[wasm_bindgen]
impl NdarrayMut {
    pub fn new(input: &mut Ndarray) -> Result<NdarrayMut, JsValue> {
        match &mut input.0 {
            NdarrayUnion::I32(ndarray) => match Rc::get_mut(&mut ndarray.data) {
                Some(mut_ref) => Ok(NdarrayMut(NdarrayUnionMut::I32(NdarrayBaseMut {
                    data: mut_ref,
                    shape: ndarray.shape.clone(),
                    strides: ndarray.strides.clone(),
                }))),
                None => Err(JsValue::from_str(
                    "Data must have single owner to be mutated",
                )),
            },
            NdarrayUnion::F64(ndarray) => match Rc::get_mut(&mut ndarray.data) {
                Some(mut_ref) => Ok(NdarrayMut(NdarrayUnionMut::F64(NdarrayBaseMut {
                    data: mut_ref,
                    shape: ndarray.shape.clone(),
                    strides: ndarray.strides.clone(),
                }))),
                None => Err(JsValue::from_str(
                    "Data must have single owner to be mutated",
                )),
            },
        }
    }
}
