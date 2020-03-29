use std::convert::TryInto;
use std::rc::Rc;

use js_sys;
use wasm_bindgen::prelude::*;

use crate::js_interop;

/// N-dimensional arary for numerical computations in javascript.
///
/// Ndarray is a n-dimensional container for homogeneous data. It enables efficient manipulation
/// of data without moving the data in memory.
///
/// # Example
///
#[wasm_bindgen]
pub struct Ndarray(NdarrayUnion);

/// Union of multiple `NdarrayBase<T>` types with different type parameters `T`.
///
/// NdarrayUnion is used to allow Ndarray to be implemented with different type parameters. This is necesarry because wasm doesn't allow generics.
///
/// # Example
///
pub enum NdarrayUnion {
    I32(NdarrayBase<i32>),
    F64(NdarrayBase<f64>),
}

/// The actual implementation of the strided n-dimensional array.
///
/// NdarrayBase enables an efficient acces of n-dimensional data stored in a contigious memory section.
///
/// # Example
///
pub struct NdarrayBase<T> {
    pub data: Rc<[T]>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub format: Format,
}

/// Describes if a `Ndarray` behaves like a contigious subview (`Slice`), a set along selected axis (`Pick`) or normally (`None`)
///
/// Subview is used because traits cannot be used at the wasm boundary. To still represent Ndarray as a polymorphic type, its behavior changes according to the value of its Subview field.
pub enum Format {
    Slice(Vec<usize>),
    Slices(Vec<Vec<usize>>),
    None,
}

/// Struct used for iterators that contains shared references to an ArrayBase object
pub struct NdarrayView<'a, T> {
    pub data: &'a [T],
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub format: FormatView<'a>,
    pub len: usize,
}

pub enum FormatView<'a> {
    Slice(&'a [usize]),
    Slices(&'a [Vec<usize>]),
    None,
}

#[wasm_bindgen]
pub struct NdarrayMut(NdarrayUnionMut);

pub enum NdarrayUnionMut {
    I32(NdarrayBaseMut<i32>),
    F64(NdarrayBaseMut<f64>),
}

pub struct NdarrayBaseMut<T> {
    pub data: *mut [T],
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub format: Format,
}

/// Struct used for iterators that contains mutable references to an ArrayBase object
pub struct NdarrayViewMut<'a, T> {
    pub data: &'a mut [T],
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub format: FormatView<'a>,
    pub len: usize,
}

#[wasm_bindgen]
impl Ndarray {
    /// Constructor for the Ndarray struct.
    ///
    /// Can be constucted with:
    /// - a nested Javascript Array
    /// - a linear Int32Array
    /// - a linear Float64Array
    ///
    /// Example:
    ///
    #[wasm_bindgen(constructor)]
    pub fn new(input: JsValue) -> Ndarray {
        match js_interop::unwrap_js_value(input) {
            Ok(data) => match data {
                js_interop::JsType::Array(array) => {
                    let mut shape: Vec<usize> = Vec::new();
                    shape.push(array.length().try_into().unwrap());
                    let flat_array = js_interop::flatten_jsarray(array, &mut shape);
                    let data: Rc<[f64]> = flat_array.iter().map(|x| x.as_f64().unwrap()).collect();
                    Ndarray(NdarrayUnion::F64(NdarrayBase {
                        data: data,
                        strides: Ndarray::get_strides_from_shape(&shape),
                        shape: shape,
                        format: Format::None,
                    }))
                }
                js_interop::JsType::Int32Array(array) => {
                    let vec = array.to_vec();
                    Ndarray(NdarrayUnion::I32(NdarrayBase {
                        shape: vec![vec.len()],
                        data: Rc::from(vec),
                        strides: vec![1],
                        format: Format::None,
                    }))
                }
                js_interop::JsType::Float64Array(array) => {
                    let vec = array.to_vec();
                    Ndarray(NdarrayUnion::F64(NdarrayBase {
                        shape: vec![vec.len()],
                        data: Rc::from(vec),
                        strides: vec![1],
                        format: Format::None,
                    }))
                }
                _ => panic!("Input must be some kind of Array."),
            },
            Err(_) => panic!("Datatype not supported"),
        }
    }

    /// Changes the shape of the given Ndarray.
    ///
    /// Length of the provided Array must be equal to the number of dimensions.
    pub fn reshape(&mut self, shape: &js_sys::Array) -> Result<(), JsValue> {
        let vec = js_interop::into_vec_usize(shape)?;
        if vec.iter().product::<usize>() == self.shape().iter().product::<usize>() {
            self.set_strides(Ndarray::get_strides_from_shape(&vec));
            Ok(())
        } else {
            Err(JsValue::from_str("Shape doesn't fit data."))
        }
    }

    /// Returns a single entry with the indices given through a Javascript Array
    pub fn get(&self, input: js_sys::Array) -> Result<JsValue, JsValue> {
        // TODO: introduce bound checks
        assert_eq!(js_sys::Array::is_array(&input.get(0)), false);
        let indices = js_interop::into_vec_isize(&input)?;
        let shape: &Vec<usize> = self.shape();
        let indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .map(|(i, x)| ((x + shape[i] as isize) % (shape[i] as isize)) as usize)
            .collect::<Vec<usize>>();
        assert_eq!(self.strides().len(), indices.len());
        let index = match self.format() {
            Format::None => self
                .strides()
                .iter()
                .zip(indices.iter())
                .map(|(x, y)| x * y)
                .sum::<usize>(),

            Format::Slice(offset) => self
                .strides()
                .iter()
                .zip(indices.iter())
                .enumerate()
                .map(|(i, (x, y))| *x * (offset[i] + y))
                .sum::<usize>(),
            Format::Slices(slices) => self
                .strides()
                .iter()
                .zip(slices.iter())
                .enumerate()
                .map(|(i, (x, y))| x * y[indices[i]])
                .sum::<usize>(),
        };
        match &self.0 {
            NdarrayUnion::I32(ndarray) => Ok(JsValue::from_f64(ndarray.data[index] as f64)),
            NdarrayUnion::F64(ndarray) => Ok(JsValue::from_f64(ndarray.data[index])),
        }
    }

    /// Creates a slice of the corresponding Ndarray. Returns a new Ndarray which still references the same memory as the original, but has potentially a different offset, shape and strides.
    ///
    /// Because the slice references the original memory, it should be used for computations but it shouldn't be assigned to a new variable. Assigning to a new variable increases the reference count and the underlying data can not be mutated afterwards. Weblab-ndarray doesn't allow mutating data if the reference count of a value is higher than one.
    ///
    /// # Example
    ///
    pub fn slice(&self, input: js_sys::Array) -> Result<Ndarray, JsValue> {
        let input = input.to_vec();
        let strides = self.strides();
        // iterator of tuple can be split into a pair of two collections
        let (offset, rest) = input
            // iterate over all slice defintions
            .iter()
            .enumerate()
            .map(|(i, x)| {
                // create vector of isizes from Array of JsValues
                match js_interop::into_vec_isize(&js_sys::Array::from(x)) {
                    // match the length of each slice definition
                    Ok(array) => match array.len() {
                        // if it has two entries, the offset and the shape changes, but not the strides
                        2 => Ok((
                            array[0] as usize,
                            ((array[1] - array[0] + 1) as usize, strides[i]),
                        )),
                        // if it has 3 entries, offset, shape and strides change
                        3 => Ok((
                            array[0] as usize,
                            (
                                (array[1] - array[0] + 1) as usize,
                                (array[2] as usize) * strides[i],
                            ),
                        )),
                        // Slice is wrongly defined
                        _ => Err(JsValue::from_str(
                            "Not the right number of entries in slice definition.",
                        )),
                    },
                    Err(err) => Err(err),
                }
            })
            // collect first into result to see of any of the slices was wrongly defined, ? Returns Err if it did, otherwise continue with vector
            .collect::<Result<Vec<(usize, (usize, usize))>, JsValue>>()?
            .into_iter()
            .unzip::<usize, (usize, usize), Vec<usize>, Vec<(usize, usize)>>();
        // unzip the second pair
        let (shape, strides): (Vec<usize>, Vec<usize>) = rest
            .into_iter()
            .unzip::<usize, usize, Vec<usize>, Vec<usize>>();
        match &self.0 {
            NdarrayUnion::I32(ndarray) => Ok(Ndarray(NdarrayUnion::I32(NdarrayBase {
                data: ndarray.data.clone(),
                strides: strides,
                shape: shape,
                format: Format::Slice(offset),
            }))),
            NdarrayUnion::F64(ndarray) => Ok(Ndarray(NdarrayUnion::F64(NdarrayBase {
                data: ndarray.data.clone(),
                strides: strides,
                shape: shape,
                format: Format::Slice(offset),
            }))),
        }
    }

    pub fn slices(&self, input: js_sys::Array) -> Result<Ndarray, JsValue> {
        let input = input.to_vec();
        let input = input
            .iter()
            .map(|x| -> Result<Vec<usize>, JsValue> {
                js_interop::into_vec_usize(&js_sys::Array::from(&x))
            })
            .collect::<Result<Vec<Vec<usize>>, JsValue>>()?;
        let shape = input.iter().map(|x| x.len()).collect::<Vec<usize>>();
        match &self.0 {
            NdarrayUnion::I32(ndarray) => Ok(Ndarray(NdarrayUnion::I32(NdarrayBase {
                data: ndarray.data.clone(),
                strides: self.strides().clone(),
                shape: shape,
                format: Format::Slices(input),
            }))),
            NdarrayUnion::F64(ndarray) => Ok(Ndarray(NdarrayUnion::F64(NdarrayBase {
                data: ndarray.data.clone(),
                strides: self.strides().clone(),
                shape: shape,
                format: Format::Slices(input),
            }))),
        }
    }
}

impl Ndarray {
    fn new_rust(input: Vec<f64>, shape: Vec<usize>) -> Result<Ndarray, String> {
        Ok(Ndarray(NdarrayUnion::F64(NdarrayBase {
            strides: Ndarray::get_strides_from_shape(&shape),
            shape: shape,
            data: Rc::from(input),
            format: Format::None,
        })))
    }

    fn view_rust<'a>(&'a self) -> Result<NdarrayView<'a, f64>, String> {
        match &self.0 {
            NdarrayUnion::F64(ndarray) => Ok(NdarrayView {
                data: &ndarray.data,
                len: ndarray.shape.iter().product::<usize>(),
                shape: &ndarray.shape,
                strides: &ndarray.strides,
                format: FormatView::None,
            }),
            NdarrayUnion::I32(_) => Err(String::from("Not the right Data")),
        }
    }
    /// Calculates the strides from a given shape.
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
    /// Returns the field `strides` from a Ndarray.
    pub fn strides(&self) -> &Vec<usize> {
        match &self.0 {
            NdarrayUnion::I32(ndarray) => &ndarray.strides,
            NdarrayUnion::F64(ndarray) => &ndarray.strides,
        }
    }
    /// Sets the field `strides` of the Ndarray according to the input.
    pub fn set_strides(&mut self, strides: Vec<usize>) {
        match &mut self.0 {
            NdarrayUnion::I32(ndarray) => ndarray.strides = strides,
            NdarrayUnion::F64(ndarray) => ndarray.strides = strides,
        }
    }
    /// Return the field `shape` of an array.
    pub fn shape(&self) -> &Vec<usize> {
        match &self.0 {
            NdarrayUnion::I32(ndarray) => &ndarray.shape,
            NdarrayUnion::F64(ndarray) => &ndarray.shape,
        }
    }
    /// Return the field `shape` of an array.
    pub fn format(&self) -> &Format {
        match &self.0 {
            NdarrayUnion::I32(ndarray) => &ndarray.format,
            NdarrayUnion::F64(ndarray) => &ndarray.format,
        }
    }
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
                    format: Format::None,
                }))),
                None => Err(JsValue::from_str(
                    "Data must have single owner to be mutated.",
                )),
            },
            NdarrayUnion::F64(ndarray) => match Rc::get_mut(&mut ndarray.data) {
                Some(mut_ref) => Ok(NdarrayMut(NdarrayUnionMut::F64(NdarrayBaseMut {
                    data: mut_ref,
                    shape: ndarray.shape.clone(),
                    strides: ndarray.strides.clone(),
                    format: Format::None,
                }))),
                None => Err(JsValue::from_str(
                    "Data must have single owner to be mutated.",
                )),
            },
        }
    }
}
