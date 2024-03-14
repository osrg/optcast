/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::fmt::Debug;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use half::f16;
use log::info;
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

#[derive(Parser, Debug, Clone)]
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

    #[arg(long, default_value = "1048576")]
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

pub(crate) fn print_stat(args: &Args, elapsed: &Duration) {
    let nsplit = args.address.split(",").count();
    let size = args.count
        * if args.data_type == DataType::F32 {
            4
        } else {
            2
        };
    let size = if args.ring_rank > 0 {
        info!(
            "type: ring, nchannel: {}, nsplit: {}, nreq: {}, nrank: {}, reduce_ths: {}, count: {}, try_count: {} #",
            args.nchannel, nsplit, args.nreq, args.nrank, args.reduce_threads, args.count, args.try_count
        );
        args.nrank * size
    } else {
        info!(
            "type: agg, nchannel: {}, nsplit: {}, nreq: {}, count: {}, try_count: {} #",
            args.nchannel, nsplit, args.nreq, args.count, args.try_count
        );
        nsplit * size
    };
    let latency = elapsed.as_micros() / args.try_count as u128;
    let size = size as f64; // B
    let latency = latency as f64 / 1000.0 / 1000.0; // s
    let bandwidth = (size * 8.0) / latency; // bps
    let bandwidth = bandwidth / 1024.0 / 1024.0 / 1024.0; // Gbps
    info!(
        "size: {:.2}MB, bandwidth: {:.2}Gbps #",
        size / 1024.0 / 1024.0,
        bandwidth
    );
}

pub(crate) fn vec_of_none<T>(n: usize) -> Vec<Option<T>> {
    std::iter::repeat_with(|| None).take(n).collect()
}

#[cfg(test)]
pub mod tests {
    use std::sync::Once;
    use crate::nccl_net;

    static INIT: Once = Once::new();

    pub(crate) fn initialize() {
        INIT.call_once(|| {
            env_logger::init();
            std::env::set_var("NCCL_PLUGIN_P2P", "socket");
            nccl_net::init();
        });
    }
}
