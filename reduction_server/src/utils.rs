/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::fmt::Debug;

use aligned_box::AlignedBox;
use clap::{Parser, ValueEnum};
use half::f16;
use half::slice::HalfFloatSliceExt;
use log::trace;
use num_traits::FromPrimitive;

pub(crate) fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum DataType {
    F32,
    F16,
}

#[derive(Parser, Debug)]
pub(crate) struct Args {
    #[arg(short, long)]
    pub verbose: bool,

    #[arg(short, long)]
    pub client: bool,

    #[arg(short, long)]
    pub bench: bool,

    #[arg(short, long, default_value = "8918")]
    pub port: u16,

    #[arg(short, long, default_value = "0.0.0.0")]
    pub address: String,

    #[arg(long, default_value = "1024")]
    pub count: usize,

    #[arg(long, default_value = "100")]
    pub try_count: usize,

    #[arg(long, default_value = "2", help = "threads per reduce job")]
    pub reduce_threads: usize,

    #[arg(long, default_value = "2")]
    pub reduce_jobs: usize,

    #[arg(long, default_value = "0")] // 0: = nrank
    pub recv_threads: usize,

    #[arg(long, default_value = "0")] // 0: = nrank
    pub send_threads: usize,

    #[arg(long, default_value = "1")]
    pub nchannel: usize,

    #[arg(long, default_value = "1")]
    pub nreq: usize,

    #[arg(long, default_value = "1")]
    pub nrank: usize,

    #[arg(long, default_value = "f32")]
    pub data_type: DataType,

    #[arg(long, default_value = "0")]
    pub ring_rank: usize,
}

pub(crate) trait Float:
    num_traits::Float + FromPrimitive + Default + Sync + Send + std::fmt::Debug + std::ops::AddAssign
{
}

impl Float for f32 {}
impl Float for f16 {}

pub(crate) fn alignment(size: usize) -> usize {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    (size + page - 1) & !(page - 1)
}

pub(crate) fn print_stat(size: usize, latency: u128) {
    let size = size as f64; // B
    let latency = latency as f64 / 1000.0 / 1000.0; // s
    let bandwidth = (size * 8.0) / latency; // bps
    let bandwidth = bandwidth / 1024.0 / 1024.0 / 1024.0; // Gbps
    trace!(
        "size: {:.2}MB, bandwidth: {:.2}Gbps",
        size / 1024.0 / 1024.0,
        bandwidth
    );
}

pub(crate) struct WorkingMemory {
    recv_bufs: Vec<AlignedBox<[f32]>>,
    send_buf: AlignedBox<[f32]>,
}

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
    fn reduce_one(&mut self, recv_buf: &[T], _: Option<&mut WorkingMemory>) -> Result<(), ()>;
}

impl<T: Float> Reduce<T> for [T] {
    default fn reduce(&mut self, _: &Vec<&[T]>, _: Option<&mut WorkingMemory>) -> Result<(), ()> {
        Err(())
    }

    default fn reduce_one(&mut self, _: &[T], _: Option<&mut WorkingMemory>) -> Result<(), ()> {
        Err(())
    }
}

impl Reduce<f16> for [f16] {
    fn reduce(
        &mut self,
        recv_bufs: &Vec<&[f16]>,
        work_mem: Option<&mut WorkingMemory>,
    ) -> Result<(), ()> {
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
        Ok(())
    }

    fn reduce_one(
        &mut self,
        recv_buf: &[f16],
        work_mem: Option<&mut WorkingMemory>,
    ) -> Result<(), ()> {
        let work_mem = work_mem.unwrap();
        recv_buf.convert_to_f32_slice(&mut work_mem.recv_bufs[0].as_mut());
        work_mem
            .send_buf
            .reduce_one(&work_mem.recv_bufs[0].as_ref(), None)?;
        self.as_mut()
            .convert_from_f32_slice(&work_mem.send_buf.as_ref());
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

    fn reduce_one(&mut self, recv_buf: &[f32], _: Option<&mut WorkingMemory>) -> Result<(), ()> {
        let (_, send, _) = self.as_simd_mut::<4>();
        let (_, recv, _) = recv_buf.as_ref().as_simd::<4>();
        for j in 0..send.len() {
            send[j] += recv[j];
        }
        Ok(())
    }
}
