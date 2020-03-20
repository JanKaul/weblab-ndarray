use std::convert::TryInto;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Flattens a given JsArray.
pub fn flatten_jsarray(input: js_sys::Array, shape: &mut Vec<usize>) -> js_sys::Array {
    if js_sys::Array::is_array(&input.get(0)) {
        shape.push(input.length().try_into().unwrap());
        flatten_jsarray(input.flat(1), shape)
    } else {
        shape.push(input.length().try_into().unwrap());
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
