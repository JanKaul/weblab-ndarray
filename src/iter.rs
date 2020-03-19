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
    ndarray: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for IterNdarrayBase<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.ndarray.next()
    }
}

impl<'a, T> IntoIterator for &'a NdarrayBase<T> {
    type Item = &'a T;
    type IntoIter = IterNdarrayBase<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        IterNdarrayBase {
            ndarray: self.data().iter(),
        }
    }
}

#[derive(Debug)]
pub struct IterMutNdarrayBase<'a, T> {
    ndarray: std::slice::IterMut<'a, T>,
}

impl<'a, T> Iterator for IterMutNdarrayBase<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        self.ndarray.next()
    }
}

impl<'a, T> IntoIterator for &'a mut NdarrayBase<T> {
    type Item = &'a mut T;
    type IntoIter = IterMutNdarrayBase<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        IterMutNdarrayBase {
            ndarray: Rc::get_mut(self.data_mut()).unwrap().iter_mut(),
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
