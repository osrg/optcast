use aligned_box::AlignedBox;
use std::ops::{Deref, DerefMut};
use std::{
    mem::forget,
    sync::{Mutex, MutexGuard},
};
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub(crate) enum Error {
    InvalidPartitionSize,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Error::InvalidPartitionSize => write!(f, "Invalid partition size"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) struct PartitionedVec<'a, T> {
    pub(crate) parts: Vec<Mutex<&'a mut [T]>>,
    ptr: *mut [T],
    layout: std::alloc::Layout,
    size: usize,
}

impl<'a, T: Debug> std::fmt::Debug for PartitionedVec<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let buf = unsafe { AlignedBox::from_raw_parts(self.ptr, self.layout) };
        let result = f.write_fmt(format_args!("{:?}", &*buf));
        forget(buf);
        result
    }
}

unsafe impl<'a, T> Send for PartitionedVec<'a, T> where T: Send {}
unsafe impl<'a, T> Sync for PartitionedVec<'a, T> where T: Sync {}

impl<'a, T> Drop for PartitionedVec<'a, T> {
    fn drop(&mut self) {
        let buf = unsafe { AlignedBox::from_raw_parts(self.ptr, self.layout) };
        drop(buf);
    }
}

pub(crate) struct Guard<'a, 'b, T> {
    vec: &'a PartitionedVec<'b, T>,
    _mutexes: Vec<MutexGuard<'a, &'b mut [T]>>,
}

impl<'a, T: Default + Copy + Clone> PartitionedVec<'a, T> {
    pub(crate) fn from_value(alignment: usize, size: usize, num_partition: usize, value: T) -> Result<PartitionedVec<'a, T>, Error> {
        // check if size is divisible by num_partition if not return an error
        if size % num_partition != 0 {
            return Err(Error::InvalidPartitionSize);
        }

        let (ptr, layout) = AlignedBox::into_raw_parts(
            AlignedBox::<[T]>::slice_from_value(alignment, size, value).unwrap(),
        );
        let parts = (0..num_partition)
            .map(|i| {
                let ptr: *mut T = ptr.cast();
                let start = i * size / num_partition;
                Mutex::new(unsafe {
                    std::slice::from_raw_parts_mut(ptr.add(start), size / num_partition)
                })
            })
            .collect::<Vec<_>>();
        Ok(PartitionedVec {
            parts,
            ptr,
            layout,
            size,
        })
    }


    pub(crate) fn new(alignment: usize, size: usize, num_partition: usize) -> Result<PartitionedVec<'a, T>, Error> {
        // check if size is divisible by num_partition if not return an error
        if size % num_partition != 0 {
            return Err(Error::InvalidPartitionSize);
        }

        let (ptr, layout) = AlignedBox::into_raw_parts(
            AlignedBox::<[T]>::slice_from_default(alignment, size).unwrap(),
        );
        let parts = (0..num_partition)
            .map(|i| {
                let ptr: *mut T = ptr.cast();
                let start = i * size / num_partition;
                Mutex::new(unsafe {
                    std::slice::from_raw_parts_mut(ptr.add(start), size / num_partition)
                })
            })
            .collect::<Vec<_>>();
        Ok(PartitionedVec {
            parts,
            ptr,
            layout,
            size,
        })
    }

    pub(crate) fn lock<'b>(&'b self) -> Guard<'b, 'a, T> {
        let _mutexes = self.parts.iter().map(|m| m.lock().unwrap()).collect();
        Guard { vec: self, _mutexes }
    }
}

impl<T> Deref for Guard<'_, '_, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        let ptr: *const T = self.vec.ptr.cast();
        unsafe { std::slice::from_raw_parts(ptr, self.vec.size) }
    }
}

impl<T> DerefMut for Guard<'_, '_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr: *mut T = self.vec.ptr.cast();
        unsafe { std::slice::from_raw_parts_mut(ptr, self.vec.size) }
    }
}

// test
#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    #[test]
    fn test_partitioned_vec() {
        let vec = Arc::new(PartitionedVec::<i32>::new(4, 4, 4).unwrap());
        {
            let vec = vec.clone();
            std::thread::spawn(move || {
                for _ in 0..10 {
                    {
                        let vec = vec.lock();
                        println!("{:?}", &*vec);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            });
        }

        (0..4)
            .map(|i| {
                let vec = vec.clone();
                std::thread::spawn(move || {
                    for _ in 0..10 {
                        {
                            let mut guard = vec.parts[i].lock().unwrap();
                            guard[0] += 1;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());

        let vec = vec.lock();
        assert_eq!(&*vec, &[10, 10, 10, 10]);
    }
}
