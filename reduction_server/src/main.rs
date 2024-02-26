/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

#![feature(c_variadic)]
#![feature(portable_simd)]
#![feature(min_specialization)]

use clap::Parser;

mod nccl_net;
mod utils;
mod partitioned_vec;
mod client;
mod server;
mod ring;

use utils::Args;
use server::server;
use client::{client, bench};
use ring::ring;

fn main() {
    let mut builder = env_logger::Builder::from_default_env();
    builder
        .target(env_logger::Target::Stdout)
        .format_timestamp_nanos()
        .init();
    nccl_net::init();

    let mut args = Args::parse();

    if args.recv_threads == 0 {
        args.recv_threads = args.nrank
    }

    if args.send_threads == 0 {
        args.send_threads = args.nrank
    }

    if args.client {
        client(args);
        return;
    } else if args.bench {
        bench(args);
        return;
    } else if args.ring_rank > 0 {
        ring(args);
        return;
    } else {
        server(args);
        return;
    }
}
