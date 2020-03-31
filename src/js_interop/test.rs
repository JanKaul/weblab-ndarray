#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use crate::js_interop::*;

wasm_bindgen_test_configure!(run_in_browser);
#[wasm_bindgen_test]
fn test_vec_isize_into_array() {
    let vec: Vec<isize> = (1..17).collect();
    let jsarray = vec_isize_into_array(vec.clone());

    assert_eq!(vec[10], jsarray.get(10).as_f64().unwrap() as isize);
}

#[wasm_bindgen_test]
fn test_vec_f64_into_float64array() {
    let vec: Vec<f64> = (1..26).into_iter().map(|x| x as f64).collect();
    let jsarray = vec_f64_into_float64array(vec.clone());

    assert_eq!(vec[14], jsarray.get_index(14));
}

#[wasm_bindgen_test]
fn test_into_vec_isize() {
    let jsarray = vec_isize_into_array((1..8).collect());

    let vec = match into_vec_isize(&jsarray) {
        Err(err) => panic!(),
        Ok(data) => data,
    };

    assert_eq!(jsarray.get(6).as_f64().unwrap() as isize, vec[6]);
}

#[wasm_bindgen_test]
fn test_into_vec_usize() {
    let jsarray = vec_isize_into_array((1..8).collect());

    let vec = match into_vec_usize(&jsarray) {
        Err(err) => panic!(),
        Ok(data) => data,
    };

    assert_eq!(jsarray.get(6).as_f64().unwrap() as usize, vec[6]);
}
