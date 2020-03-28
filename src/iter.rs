use crate::ndarray::*;
use std::rc::Rc;

pub struct ViewIter<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    format: FormatView<'a>,
    count: usize,
    len: usize,
}

impl<'a, T> Iterator for ViewIter<'a, T> {
    type Item = NdarrayView<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.shape[0] {
            let count = self.count;
            self.count += 1;
            match self.format {
                FormatView::None => Some(NdarrayView {
                    data: unsafe {
                        std::slice::from_raw_parts(
                            &*(&self.data[0] as *const T)
                                .offset(((count) * self.strides[0]) as isize),
                            self.len,
                        )
                    },
                    shape: &self.shape[1..],
                    strides: &self.strides[1..],
                    format: FormatView::None,
                }),
                FormatView::Slice(offset) => Some(NdarrayView {
                    data: unsafe {
                        std::slice::from_raw_parts(
                            &*(&self.data[0] as *const T)
                                .offset(((offset[0] + count) * self.strides[0]) as isize),
                            self.len,
                        )
                    },
                    shape: &self.shape[1..],
                    strides: &self.strides[1..],
                    format: FormatView::Slice(&offset[1..]),
                }),
                FormatView::Slices(slices) => Some(NdarrayView {
                    data: unsafe {
                        std::slice::from_raw_parts(
                            &*(&self.data[0] as *const T)
                                .offset(((slices[0][count]) * self.strides[0]) as isize),
                            self.len,
                        )
                    },
                    shape: &self.shape[1..],
                    strides: &self.strides[1..],
                    format: FormatView::Slices(&slices[1..]),
                }),
            }
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
        match self.format {
            FormatView::None => ViewIter {
                data: &self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::None,
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            FormatView::Slice(offset) => ViewIter {
                data: &self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slice(&offset),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            FormatView::Slices(slices) => ViewIter {
                data: &self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slices(&slices),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
        }
    }
}

impl<'a, T> IntoIterator for &'a NdarrayBase<T> {
    type Item = NdarrayView<'a, T>;
    type IntoIter = ViewIter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        match &self.format {
            Format::None => ViewIter {
                data: &self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::None,
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            Format::Slice(offset) => ViewIter {
                data: &self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slice(&offset),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            Format::Slices(slices) => ViewIter {
                data: &self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slices(&slices),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
        }
    }
}

pub struct ViewIterMut<'a, T> {
    pub data: &'a mut [T],
    shape: &'a [usize],
    strides: &'a [usize],
    format: FormatView<'a>,
    count: usize,
    len: usize,
}

impl<'a, T> Iterator for ViewIterMut<'a, T> {
    type Item = NdarrayViewMut<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.shape[0] {
            let count = self.count;
            self.count += 1;
            match self.format {
                FormatView::None => Some(NdarrayViewMut {
                    data: unsafe {
                        std::slice::from_raw_parts_mut(
                            &mut *(&mut self.data[0] as *mut T)
                                .offset(((count) * self.strides[0]) as isize),
                            self.len,
                        )
                    },
                    shape: &self.shape[1..],
                    strides: &self.strides[1..],
                    format: FormatView::None,
                }),
                FormatView::Slice(offset) => Some(NdarrayViewMut {
                    data: unsafe {
                        std::slice::from_raw_parts_mut(
                            &mut *(&mut self.data[0] as *mut T)
                                .offset(((offset[0] + count) * self.strides[0]) as isize),
                            self.len,
                        )
                    },
                    shape: &self.shape[1..],
                    strides: &self.strides[1..],
                    format: FormatView::Slice(&offset[1..]),
                }),
                FormatView::Slices(slices) => Some(NdarrayViewMut {
                    data: unsafe {
                        std::slice::from_raw_parts_mut(
                            &mut *(&mut self.data[0] as *mut T)
                                .offset(((slices[0][count]) * self.strides[0]) as isize),
                            self.len,
                        )
                    },
                    shape: &self.shape[1..],
                    strides: &self.strides[1..],
                    format: FormatView::Slices(&slices[1..]),
                }),
            }
        // Some(NdarrayViewMut {
        //     data: unsafe {
        //         std::slice::from_raw_parts_mut(
        //             &mut *(&mut self.data[0] as *mut T)
        //                 .offset(((count) * self.strides[0]) as isize),
        //             self.len,
        //         )
        //     },
        //     shape: &self.shape[1..],
        //     strides: &self.strides[1..],
        // })
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
        match self.format {
            FormatView::None => ViewIterMut {
                data: &mut self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::None,
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            FormatView::Slice(offset) => ViewIterMut {
                data: &mut self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slice(&offset),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            FormatView::Slices(slices) => ViewIterMut {
                data: &mut self.data,
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slices(&slices),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
        }
    }
}

impl<'a, T> IntoIterator for &'a mut NdarrayBase<T> {
    type Item = NdarrayViewMut<'a, T>;
    type IntoIter = ViewIterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        match &self.format {
            Format::None => ViewIterMut {
                data: Rc::get_mut(&mut self.data).unwrap(),
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::None,
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            Format::Slice(offset) => ViewIterMut {
                data: Rc::get_mut(&mut self.data).unwrap(),
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slice(&offset),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
            Format::Slices(slices) => ViewIterMut {
                data: Rc::get_mut(&mut self.data).unwrap(),
                shape: &self.shape,
                strides: &self.strides,
                format: FormatView::Slices(&slices),
                count: 0,
                len: self.shape[1..].iter().product(),
            },
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
