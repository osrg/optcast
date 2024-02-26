/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use half::f16;
use log::info;

use crate::utils::*;

use crate::nccl_net;
use crate::nccl_net::{Comm, Request};

use crate::partitioned_vec::PartitionedVec;

fn do_ring<T: Float>(args: &Args, recv: Comm, send: Comm) {
    let size = args.count * std::mem::size_of::<T>();
    let initial: T = T::from_usize(args.ring_rank).unwrap();

    let tag = 0x69;

    let acc = PartitionedVec::<T>::from_value(
        alignment(size),
        args.count * args.nrank,
        args.nrank,
        initial,
    )
    .unwrap();

    let init = vec![initial; args.count * args.nrank];

    let buf =
        PartitionedVec::<T>::new(alignment(size), args.count * args.nrank, args.nrank).unwrap();

    let mhs = (0..args.nrank)
        .map(|i| {
            let rh = nccl_net::reg_mr(&recv, &mut buf.parts[i].lock().unwrap()).unwrap();
            let sh = nccl_net::reg_mr(&send, &mut acc.parts[i].lock().unwrap()).unwrap();
            (rh, sh)
        })
        .collect::<Vec<_>>();

    // start timer
    let mut start = std::time::Instant::now();

    for c in 0..args.try_count {
        if c == 1 {
            info!("start timer");
            start = std::time::Instant::now(); // restart timer
        }
        acc.lock().copy_from_slice(&init);

        (0..args.nrank - 1).for_each(|i| {
            let mut sreq: Option<Request> = None;
            let mut rreq: Option<Request> = None;

            let sidx = (args.nrank + args.ring_rank + i - 1) % args.nrank;
            let ridx = (args.nrank + args.ring_rank + i) % args.nrank;

            loop {
                if sreq.is_none() {
                    sreq =
                        nccl_net::isend(&send, &mhs[sidx].1, &acc.parts[sidx].lock().unwrap(), tag)
                            .unwrap();
                }
                if rreq.is_none() {
                    rreq = nccl_net::irecv(
                        &recv,
                        &mhs[ridx].0,
                        &mut buf.parts[ridx].lock().unwrap(),
                        tag,
                    )
                    .unwrap();
                }
                if sreq.is_some() && rreq.is_some() {
                    break;
                }
            }

            loop {
                if sreq.is_some() {
                    let (done, _) = nccl_net::test(&sreq.as_ref().unwrap()).unwrap();
                    if done {
                        sreq = None;
                    }
                }
                if rreq.is_some() {
                    let (done, _) = nccl_net::test(&rreq.as_ref().unwrap()).unwrap();
                    if done {
                        rreq = None;
                    }
                }
                if sreq.is_none() && rreq.is_none() {
                    break;
                }
            }

            // do reduction
            {
                let mut acc = acc.parts[ridx].lock().unwrap();
                let buf = buf.parts[ridx].lock().unwrap();
                for i in 0..acc.len() {
                    acc[i] += buf[i];
                }
            }
        });

        (0..args.nrank - 1).for_each(|i| {
            let mut sreq: Option<Request> = None;
            let mut rreq: Option<Request> = None;

            let sidx = (args.nrank - 1 + args.nrank + args.ring_rank + i - 1) % args.nrank;
            let ridx = (args.nrank - 1 + args.nrank + args.ring_rank + i) % args.nrank;

            loop {
                if sreq.is_none() {
                    sreq =
                        nccl_net::isend(&send, &mhs[sidx].1, &acc.parts[sidx].lock().unwrap(), tag)
                            .unwrap();
                }
                if rreq.is_none() {
                    rreq = nccl_net::irecv(
                        &recv,
                        &mhs[ridx].0,
                        &mut acc.parts[ridx].lock().unwrap(),
                        tag,
                    )
                    .unwrap();
                }
                if sreq.is_some() && rreq.is_some() {
                    break;
                }
            }

            loop {
                if sreq.is_some() {
                    let (done, _) = nccl_net::test(&sreq.as_ref().unwrap()).unwrap();
                    if done {
                        sreq = None;
                    }
                }
                if rreq.is_some() {
                    let (done, _) = nccl_net::test(&rreq.as_ref().unwrap()).unwrap();
                    if done {
                        rreq = None;
                    }
                }
                if sreq.is_none() && rreq.is_none() {
                    break;
                }
            }
        });
    }

    // stop timer
    let elapsed = start.elapsed();
    print_stat(
        args.count * std::mem::size_of::<T>() * args.nrank,
        elapsed.as_micros() / args.try_count as u128,
    );
}

pub(crate) fn ring(args: Args) {
    let args = Arc::new(args);

    let recvs = {
        let args = Arc::clone(&args);
        std::thread::spawn(move || {
            let listener = TcpListener::bind(format!("0.0.0.0:{}", args.port)).unwrap();
            let (mut recv, _) = listener.accept().unwrap();

            (0..args.nchannel)
                .map(|_| {
                    let (lcomm, handle) = nccl_net::listen().unwrap();
                    // send size of handle
                    let size = handle.len() as u32;
                    recv.write_all(&size.to_le_bytes()).unwrap();
                    // send handle
                    recv.write_all(&handle).unwrap();

                    loop {
                        let comm = nccl_net::accept(&lcomm).unwrap();
                        if comm.is_some() {
                            return comm.unwrap();
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
    };

    let sends = {
        let args = Arc::clone(&args);
        std::thread::spawn(move || {
            let mut send = loop {
                let res = TcpStream::connect(&args.address);
                if res.is_ok() {
                    break res.unwrap();
                }
                // sleep 1s
                std::thread::sleep(std::time::Duration::from_secs(1));
            };
            (0..args.nchannel)
                .map(|_| {
                    let mut buffer = [0u8; 4];
                    send.read(buffer.as_mut()).unwrap();
                    let size = u32::from_le_bytes(buffer);
                    let mut handle = vec![0u8; size as usize];
                    send.read(handle.as_mut()).unwrap();
                    info!("received handle: {:?}", handle);

                    loop {
                        let comm = nccl_net::connect(&handle).unwrap();
                        if comm.is_some() {
                            return comm.unwrap();
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
    };

    let recvs = recvs.join().unwrap();
    let sends = sends.join().unwrap();

    let hs = recvs
        .into_iter()
        .zip(sends.into_iter())
        .map(|(recv, send)| {
            let args = Arc::clone(&args);
            std::thread::spawn(move || {
                if args.data_type == DataType::F32 {
                    do_ring::<f32>(&args, recv, send);
                } else {
                    do_ring::<f16>(&args, recv, send);
                }
            })
        })
        .collect::<Vec<_>>();
    hs.into_iter().for_each(|h| h.join().unwrap());
}
