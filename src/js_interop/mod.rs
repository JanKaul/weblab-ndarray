use std::convert::TryInto;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

pub mod test;

/// Flattens a given JsArray.
pub fn flatten_jsarray(input: js_sys::Array, shape: &mut Vec<usize>) -> js_sys::Array {
    if js_sys::Array::is_array(&input.get(0)) {
        shape.push(
            input
                .get(0)
                .unchecked_into::<js_sys::Array>()
                .length()
                .try_into()
                .unwrap(),
        );
        flatten_jsarray(input.flat(1), shape)
    } else {
        input
    }
}

/// Supported types for conversion from javascript
pub enum JsType {
    Number(f64),
    Array(js_sys::Array),
    Int32Array(js_sys::Int32Array),
    Float64Array(js_sys::Float64Array),
}

/// Unwraps a given JsValue to a JsType
///
/// Can be unwraped into:
/// - a number
/// - an Array
/// - a Int32Array
/// - a Float64Array
pub fn unwrap_js_value(input: JsValue) -> Result<JsType, JsValue> {
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

/// Turns a Javascript Array into a `Vec<usize>`
pub fn into_vec_usize(input: &js_sys::Array) -> Result<Vec<usize>, JsValue> {
    input
        .iter()
        .into_iter()
        .map(|x: JsValue| match x.as_f64() {
            Some(n) => Ok(n as usize),
            None => Err(JsValue::from_str("Indices must be only numbers")),
        })
        .collect()
}

/// Turns a Javascript Array into a `Vec<isize>`
pub fn into_vec_isize(input: &js_sys::Array) -> Result<Vec<isize>, JsValue> {
    input
        .iter()
        .into_iter()
        .map(|x: JsValue| match x.as_f64() {
            Some(n) => Ok(n as isize),
            None => Err(JsValue::from_str("Indices must be only numbers")),
        })
        .collect()
}

pub fn vec_f64_into_float64array(input: Vec<f64>) -> js_sys::Float64Array {
    js_sys::Float64Array::from(input.as_slice())
}

pub fn vec_isize_into_array(input: Vec<isize>) -> js_sys::Array {
    input
        .into_iter()
        .map(|x| JsValue::from_f64(x as f64))
        .collect()
}

pub fn vecvec_isize_into_arrayarray(input: Vec<Vec<isize>>) -> js_sys::Array {
    input
        .into_iter()
        .map(|x| {
            x.into_iter()
                .map(|y| JsValue::from_f64(y as f64))
                .collect::<js_sys::Array>()
        })
        .collect()
}

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);

    #[wasm_bindgen(js_namespace = console, js_name = log)]
    pub fn log_usize(a: usize);
}
