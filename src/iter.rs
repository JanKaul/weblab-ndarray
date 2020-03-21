use crate::ndarray::*;
use std::rc::Rc;

impl<T> NdarrayBase<T> {
    pub fn iter<'a>(&'a self) -> IterNdarrayBase<'a, T> {
        self.into_iter()
    }
    pub fn iter_mut<'a>(&'a mut self) -> IterMutNdarrayBase<'a, T> {
        self.into_iter()
    }
}

#[derive(Debug)]
pub struct IterNdarrayBase<'a, T> {
    slice: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for IterNdarrayBase<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next()
    }
}

impl<'a, T> IntoIterator for &'a NdarrayBase<T> {
    type Item = &'a T;
    type IntoIter = IterNdarrayBase<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        IterNdarrayBase {
            slice: self.data().iter(),
        }
    }
}

#[derive(Debug)]
pub struct IterMutNdarrayBase<'a, T> {
    slice: std::slice::IterMut<'a, T>,
}

impl<'a, T> Iterator for IterMutNdarrayBase<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        self.slice.next()
    }
}

impl<'a, T> IntoIterator for &'a mut NdarrayBase<T> {
    type Item = &'a mut T;
    type IntoIter = IterMutNdarrayBase<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        IterMutNdarrayBase {
            slice: Rc::get_mut(self.data_mut()).unwrap().iter_mut(),
        }
    }
}

// use std::iter::FromIterator;
//
// impl<'a, S: Copy> FromIterator<&'a S> for NdarrayBase<S> {
//     fn from_iter<T>(iter: T) -> Self
//     where
//         T: IntoIterator<Item = &'a S>,
//     {
//         NdarrayBase {
//             data: iter.into_iter().map(|x| *x).collect(),
//         }
//     }
// }

#[derive(Debug)]
pub struct IterAxisNdarrayBase<'a, T> {
    ndarray: std::slice::Iter<'a, T>,
    axis_len: &'a usize,
    axis_stride: &'a usize,
    count: usize,
    offset: usize,
}

impl<'a, T> Iterator for IterAxisNdarrayBase<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            self.count += 1;
            self.ndarray.nth(self.offset)
        } else if self.count < *self.axis_len {
            self.count += 1;
            self.ndarray.nth(*self.axis_stride - 1)
        } else {
            None
        }
    }
}

impl<'a, T> NdarrayBase<T> {
    fn iter_axis(&'a self, axis: usize, mut index: Vec<usize>) -> IterAxisNdarrayBase<'a, T> {
        index.insert(axis, 0);
        let offset = self
            .strides()
            .iter()
            .zip(index.iter())
            .map(|(x, y)| x * y)
            .sum();
        IterAxisNdarrayBase {
            ndarray: self.data().iter(),
            axis_len: &self.shape()[axis],
            axis_stride: &self.strides()[axis],
            count: 0,
            offset: offset,
        }
    }

    fn iter_outer(&'a self, mut index: Vec<usize>) -> IterAxisNdarrayBase<'a, T> {
        index.push(0);
        let offset = self
            .strides()
            .iter()
            .zip(index.iter())
            .map(|(x, y)| x * y)
            .sum();
        IterAxisNdarrayBase {
            ndarray: self.data().iter(),
            axis_len: &self.shape().last().unwrap(),
            axis_stride: &self.strides().last().unwrap(),
            count: 0,
            offset: offset,
        }
    }
}
