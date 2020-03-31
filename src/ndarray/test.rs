#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

use crate::ndarray::*;

wasm_bindgen_test_configure!(run_in_browser);
#[wasm_bindgen_test]
fn test_new_reshape_get() {
    let input = (1..28).into_iter().map(|x| x as f64).collect::<Vec<f64>>();
    let mut ndarray = Ndarray::new(JsValue::from(js_interop::vec_f64_into_float64array(input)));
    let shape = js_interop::vec_isize_into_array(vec![3, 3, 3]);
    ndarray.reshape(&shape).unwrap();
    assert_eq!(
        ndarray
            .get(js_interop::vec_isize_into_array(vec![0, 1, 2]))
            .unwrap()
            .as_f64()
            .unwrap(),
        6.0
    );
}

#[wasm_bindgen_test]
fn test_slice() {
    let input = (1..28).into_iter().map(|x| x as f64).collect::<Vec<f64>>();
    let mut ndarray = Ndarray::new(JsValue::from(js_interop::vec_f64_into_float64array(input)));
    let shape = js_interop::vec_isize_into_array(vec![3, 3, 3]);
    ndarray.reshape(&shape).unwrap();
    match ndarray.slice(js_interop::vecvec_isize_into_arrayarray(vec![
        vec![1, 3],
        vec![0, 2],
        vec![0, 1],
    ])) {
        Err(err) => panic!(),
        Ok(slice) => {
            assert_eq!(
                slice
                    .get(js_interop::vec_isize_into_array(vec![0, 1, 0]))
                    .unwrap()
                    .as_f64()
                    .unwrap(),
                13.0
            );
            ()
        }
    }
}

#[wasm_bindgen_test]
fn test_slices() {
    let input = (1..28).into_iter().map(|x| x as f64).collect::<Vec<f64>>();
    let mut ndarray = Ndarray::new(JsValue::from(js_interop::vec_f64_into_float64array(input)));
    let shape = js_interop::vec_isize_into_array(vec![3, 3, 3]);
    ndarray.reshape(&shape).unwrap();
    match ndarray.slices(js_interop::vecvec_isize_into_arrayarray(vec![
        vec![0, 2],
        vec![0, 2],
        vec![0, 2],
    ])) {
        Err(err) => panic!(),
        Ok(slices) => {
            assert_eq!(
                slices
                    .get(js_interop::vec_isize_into_array(vec![0, 1, 0]))
                    .unwrap()
                    .as_f64()
                    .unwrap(),
                7.0
            );
            ()
        }
    }
}
