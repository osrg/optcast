/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::fmt::Debug;

use clap::{Parser, ValueEnum};
use half::f16;
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
}

pub(crate) trait Float:
    num_traits::Float + FromPrimitive + Default + Sync + Send + std::fmt::Debug
{
}

impl Float for f32 {}
impl Float for f16 {}

pub(crate) fn alignment(size: usize) -> usize {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    (size + page - 1) & !(page - 1)
}
