/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use aligned_box::AlignedBox;
use half::f16;

use crate::utils::{alignment, Float};

#[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
mod aarch64;

#[cfg(not(all(target_arch = "aarch64", target_feature = "fp16")))]
use half::slice::HalfFloatSliceExt;

#[allow(dead_code)]
pub(crate) struct WorkingMemory {
    recv_bufs: Vec<AlignedBox<[f32]>>,
    send_buf: AlignedBox<[f32]>,
}

#[allow(dead_code)]
impl WorkingMemory {
    pub(crate) fn new(count: usize, num_recv: usize) -> Self {
        let recv_bufs = (0..num_recv)
            .map(|_| AlignedBox::<[f32]>::slice_from_default(alignment(count), count).unwrap())
            .collect::<Vec<_>>();
        let send_buf = AlignedBox::<[f32]>::slice_from_default(alignment(count), count).unwrap();
        Self {
            recv_bufs,
            send_buf,
        }
    }
}

pub(crate) trait Reduce<T> {
    fn reduce(
        &mut self,
        recv_bufs: &Vec<&[T]>,
        work_mem: Option<&mut WorkingMemory>,
    ) -> Result<(), ()>;
}

impl<T: Float> Reduce<T> for [T] {
    default fn reduce(&mut self, _: &Vec<&[T]>, _: Option<&mut WorkingMemory>) -> Result<(), ()> {
        Err(())
    }
}

impl Reduce<f16> for [f16] {
    #[allow(unused_variables)]
    fn reduce(
        &mut self,
        recv_bufs: &Vec<&[f16]>,
        work_mem: Option<&mut WorkingMemory>,
    ) -> Result<(), ()> {
        cfg_if::cfg_if! {
            if #[cfg(all(
                target_arch = "aarch64",
                target_feature = "fp16"
            ))] {
                for (i, recv) in recv_bufs.iter().enumerate() {
                    if i == 0 {
                        self.copy_from_slice(recv);
                    } else {
                        unsafe {aarch64::add_assign_f16_aligned_slice(self, recv);}
                    }
                }
            }
            else {
                let work_mem = work_mem.unwrap();
                for (i, recv) in recv_bufs.iter().enumerate() {
                    recv.convert_to_f32_slice(&mut work_mem.recv_bufs[i].as_mut());
                }
                work_mem.send_buf.reduce(
                    &work_mem
                        .recv_bufs
                        .iter()
                        .map(|v| {
                            let slice_ref: &[f32] = &**v;
                            slice_ref
                        })
                        .collect(),
                    None,
                )?;
                self.as_mut()
                    .convert_from_f32_slice(&work_mem.send_buf.as_ref());
                }
        }
        Ok(())
    }
}

// impl<T: Float + std::simd::SimdElement> Reduce<T> for AlignedBox<[T]> can't compile
// error: cannot specialize on trait `SimdElement`
// --> src/main.rs:139:17
// |
// 139 | impl<T: Float + std::simd::SimdElement> Reduce<T> for AlignedBox<[T]> {
impl Reduce<f32> for [f32] {
    fn reduce(&mut self, recv_bufs: &Vec<&[f32]>, _: Option<&mut WorkingMemory>) -> Result<(), ()> {
        let (_, send, _) = self.as_simd_mut::<4>();
        for (i, recv) in recv_bufs.iter().enumerate() {
            let (_, recv, _) = recv.as_ref().as_simd::<4>();
            if i == 0 {
                send.copy_from_slice(&recv.as_ref());
            } else {
                for j in 0..send.len() {
                    send[j] += recv[j];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;

    fn bench_reduce<T>(b: &mut test::Bencher)
    where
        T: Float
            + std::fmt::Debug
            + std::ops::AddAssign
            + std::default::Default
            + std::clone::Clone,
    {
        let count = 1024;
        let num_recv = 4;
        let mut work_mem = WorkingMemory::new(count, num_recv);
        let mut recv_bufs = vec![];
        for _ in 0..num_recv {
            recv_bufs.push(
                AlignedBox::<[T]>::slice_from_value(alignment(count), count, T::default()).unwrap(),
            );
        }
        let mut send_buf =
            AlignedBox::<[T]>::slice_from_value(alignment(count), count, T::default()).unwrap();
        b.iter(|| {
            send_buf.reduce(
                &recv_bufs
                    .iter()
                    .map(|v| {
                        let slice_ref: &[T] = &**v;
                        slice_ref
                    })
                    .collect(),
                Some(&mut work_mem),
            )
        });
    }

    #[bench]
    fn bench_f16_reduce(b: &mut test::Bencher) {
        bench_reduce::<f16>(b);
    }

    #[bench]
    fn bench_f32_reduce(b: &mut test::Bencher) {
        bench_reduce::<f32>(b);
    }
}
