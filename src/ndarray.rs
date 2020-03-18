use js_sys;
use std::convert::TryInto;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use std::rc::Rc;

pub enum Data {
    I32(Rc<[i32]>),
    F64(Rc<[f64]>),
}

#[wasm_bindgen]
pub struct Ndarray {
    data: Data,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[wasm_bindgen]
impl Ndarray {
    #[wasm_bindgen(constructor)]
    pub fn new(input: JsValue) -> Result<Ndarray, JsValue> {
        match unwrap_js_value(input)? {
            JsType::Array(array) => {
                let mut shape: Vec<usize> = Vec::new();
                let flat_array = flatten_jsarray(array, &mut shape);
                let data: Rc<[f64]> = flat_array.iter().map(|x| x.as_f64().unwrap()).collect();
                Ok(Ndarray {
                    data: Data::F64(data),
                    strides: Ndarray::get_strides_from_shape(&shape),
                    shape: shape,
                })
            }
            JsType::Int32Array(array) => {
                let vec = array.to_vec();
                Ok(Ndarray {
                    shape: vec![vec.len()],
                    data: Data::I32(Rc::from(vec)),
                    strides: vec![1],
                })
            }
            JsType::Float64Array(array) => {
                let vec = array.to_vec();
                Ok(Ndarray {
                    shape: vec![vec.len()],
                    data: Data::F64(Rc::from(vec)),
                    strides: vec![1],
                })
            }
            _ => Err(JsValue::from_str("Input must be some kind of Array")),
        }
    }

    pub fn reshape(&mut self, shape: &js_sys::Array) -> Result<(), JsValue> {
        let vec = into_vec_usize(shape)?;
        if vec.iter().product::<usize>() == self.shape.iter().product::<usize>() {
            self.strides = Ndarray::get_strides_from_shape(&vec);
            Ok(())
        } else {
            Err(JsValue::from_str("Shape doesn't fit data"))
        }
    }

    pub fn get(&self, indices: Vec<usize>) -> JsValue {
        assert_eq!(self.strides.len(), indices.len());
        let index = self
            .strides
            .iter()
            .zip(indices.iter())
            .map(|(x, y)| x * y)
            .sum::<usize>();
        match &self.data {
            Data::I32(array) => JsValue::from_f64(*&array[index] as f64),
            Data::F64(array) => JsValue::from_f64(*&array[index]),
        }
    }

    pub fn slice(ndarray: &Ndarray) -> Ndarray {
        match &ndarray.data {
            Data::I32(data) => Ndarray {
                data: Data::I32(data.clone()),
                strides: ndarray.strides.clone(),
                shape: ndarray.shape.clone(),
            },
            Data::F64(data) => Ndarray {
                data: Data::F64(data.clone()),
                strides: ndarray.strides.clone(),
                shape: ndarray.shape.clone(),
            },
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

    pub fn data(&mut self) -> &mut Data {
        &mut self.data
    }
}

fn flatten_jsarray(input: js_sys::Array, shape: &mut Vec<usize>) -> js_sys::Array {
    if js_sys::Array::is_array(&input.get(0)) {
        shape.push(input.length().try_into().unwrap());
        flatten_jsarray(input.flat(1), shape)
    } else {
        shape.push(input.length().try_into().unwrap());
        input
    }
}

enum JsType {
    Number(f64),
    Array(js_sys::Array),
    Int32Array(js_sys::Int32Array),
    Float64Array(js_sys::Float64Array),
}

fn unwrap_js_value(input: JsValue) -> Result<JsType, JsValue> {
    if let Some(number) = input.as_f64() {
        Ok(JsType::Number(number))
    } else if input.is_instance_of::<js_sys::Array>() {
        Ok(JsType::Array(input.unchecked_into::<js_sys::Array>()))
    } else if input.is_instance_of::<js_sys::Int32Array>() {
        Ok(JsType::Int32Array(
            input.unchecked_into::<js_sys::Int32Array>(),
        ))
    } else if input.is_instance_of::<js_sys::Float64Array>() {
        Ok(JsType::Float64Array(
            input.unchecked_into::<js_sys::Float64Array>(),
        ))
    } else {
        Err(JsValue::from_str("JsValue type not supported"))
    }
}

fn into_vec_usize(input: &js_sys::Array) -> Result<Vec<usize>, JsValue> {
    input
        .iter()
        .into_iter()
        .map(|x: JsValue| match x.as_f64() {
            Some(n) => Ok(n as usize),
            None => Err(JsValue::from_str("Indices must be only numbers")),
        })
        .collect()
}
