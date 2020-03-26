use crate::ndarray::*;
use std::rc::Rc;

pub struct ViewIter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    count: usize,
    len: usize,
}

impl<'a, T> Iterator for ViewIter<'a, T> {
    type Item = NdarrayView<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.shape[0] {
            let count = self.count;
            self.count += 1;
            Some(NdarrayView {
                data: unsafe {
                    std::slice::from_raw_parts(
                        &*(&self.data[0] as *const T).offset(((count) * self.strides[0]) as isize),
                        self.len,
                    )
                },
                shape: &self.shape[1..],
                strides: &self.strides[1..],
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.shape[0] - self.count;
        (size, Some(size))
    }
}

impl<'a, T> IntoIterator for &'a NdarrayView<'a, T> {
    type Item = NdarrayView<'a, T>;
    type IntoIter = ViewIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ViewIter {
            data: &self.data,
            shape: &self.shape,
            strides: &self.strides,
            count: 0,
            len: self.shape[1..].iter().product(),
        }
    }
}

impl<'a, T> IntoIterator for &'a NdarrayBase<T> {
    type Item = NdarrayView<'a, T>;
    type IntoIter = ViewIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ViewIter {
            data: &self.data,
            shape: &self.shape,
            strides: &self.strides,
            count: 0,
            len: self.shape[1..].iter().product(),
        }
    }
}

pub struct ViewIterMut<'a, T> {
    pub data: &'a mut [T],
    shape: &'a [usize],
    strides: &'a [usize],
    count: usize,
    len: usize,
}

impl<'a, T> Iterator for ViewIterMut<'a, T> {
    type Item = NdarrayViewMut<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.shape[0] {
            let count = self.count;
            self.count += 1;
            Some(NdarrayViewMut {
                data: unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut *(&mut self.data[0] as *mut T)
                            .offset(((count) * self.strides[0]) as isize),
                        self.len,
                    )
                },
                shape: &self.shape[1..],
                strides: &self.strides[1..],
            })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.shape[0] - self.count;
        (size, Some(size))
    }
}

impl<'a, T> IntoIterator for &'a mut NdarrayViewMut<'a, T> {
    type Item = NdarrayViewMut<'a, T>;
    type IntoIter = ViewIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ViewIterMut {
            data: &mut self.data,
            shape: &self.shape,
            strides: &self.strides,
            count: 0,
            len: self.shape[1..].iter().product(),
        }
    }
}

impl<'a, T> IntoIterator for &'a mut NdarrayBase<T> {
    type Item = NdarrayViewMut<'a, T>;
    type IntoIter = ViewIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        ViewIterMut {
            data: Rc::get_mut(&mut self.data).unwrap(),
            shape: &self.shape,
            strides: &self.strides,
            count: 0,
            len: self.shape[1..].iter().product(),
        }
    }
}

impl<'a, T> NdarrayView<'a, T> {
    fn iter(&'a self) -> ViewIter<'a, T> {
        self.into_iter()
    }
}

impl<'a, T> NdarrayViewMut<'a, T> {
    fn iter_mut(&'a mut self) -> ViewIterMut<'a, T> {
        self.into_iter()
    }
}

impl<'a, T> NdarrayBase<T> {
    fn iter(&'a self) -> ViewIter<'a, T> {
        self.into_iter()
    }

    fn iter_mut(&'a mut self) -> ViewIterMut<'a, T> {
        self.into_iter()
    }
}
