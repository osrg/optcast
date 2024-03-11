/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use half::f16;
use log::{info, trace};

use crate::utils::*;

use crate::nccl_net;
use crate::nccl_net::{Comm, Request};

use crate::partitioned_vec::PartitionedVec;

fn do_client<T: Float>(args: &Args, comms: Vec<(Comm, Comm)>) {
    let size = args.count * std::mem::size_of::<T>();
    let initial: T = T::from_f32(2.0).unwrap();

    let tag = 0x69;

    let mut reqs = (0..args.nreq)
        .map(|i| {
            let sbuf = PartitionedVec::<T>::from_value(
                alignment(size),
                args.count * comms.len(),
                comms.len(),
                initial,
            )
            .unwrap();
            let rbuf =
                PartitionedVec::<T>::new(alignment(size), args.count * comms.len(), comms.len())
                    .unwrap();

            let mhs = comms
                .iter()
                .enumerate()
                .map(|(i, (scomm, rcomm))| {
                    let s_mhandle =
                        nccl_net::reg_mr(scomm, &mut sbuf.parts[i].lock().unwrap()).unwrap();
                    let r_mhandle =
                        nccl_net::reg_mr(rcomm, &mut rbuf.parts[i].lock().unwrap()).unwrap();
                    (s_mhandle, r_mhandle)
                })
                .collect::<Vec<_>>();

            (i, None, sbuf, rbuf, mhs)
        })
        .collect::<Vec<_>>();

    let mut finished = 0;
    let mut reqed = 0;

    // start timer
    let start = std::time::Instant::now();

    loop {
        for (i, req, sbuf, rbuf, mhs) in reqs.iter_mut() {
            if req.is_none() && reqed < args.try_count {
                *req = Some(
                    comms
                        .iter()
                        .enumerate()
                        .map(|(j, (scomm, rcomm))| {
                            let (s_mhandle, r_mhandle) = &mhs[j];
                            let mut srequest: Option<Request> = None;
                            let mut rrequest: Option<Request> = None;

                            loop {
                                if srequest.is_none() {
                                    srequest = nccl_net::isend(
                                        scomm,
                                        s_mhandle,
                                        &sbuf.parts[j].lock().unwrap(),
                                        tag,
                                    )
                                    .unwrap();
                                    if srequest.is_some() {
                                        trace!("send  : idx: {}, j: {} start", i, j);
                                    }
                                }
                                if rrequest.is_none() {
                                    rrequest = nccl_net::irecv(
                                        rcomm,
                                        r_mhandle,
                                        &mut rbuf.parts[j].lock().unwrap(),
                                        tag,
                                    )
                                    .unwrap();
                                    if srequest.is_some() {
                                        trace!("recv : idx: {}, j: {} start", i, j);
                                    }
                                }
                                if srequest.is_some() && rrequest.is_some() {
                                    break;
                                }
                            }
                            (srequest, rrequest)
                        })
                        .collect::<Vec<_>>(),
                );
                reqed += 1;
            }

            if req.is_some() {
                let mut all_done = true;
                for (j, (srequest, rrequest)) in req.as_mut().unwrap().iter_mut().enumerate() {
                    if srequest.is_some() {
                        let (send_done, _) = nccl_net::test(&srequest.as_ref().unwrap()).unwrap();
                        if send_done {
                            trace!("send  : idx: {}, j: {} done", i, j);
                            *srequest = None;
                        }
                    }
                    if rrequest.is_some() {
                        let (recv_done, _) = nccl_net::test(&rrequest.as_ref().unwrap()).unwrap();
                        if recv_done {
                            trace!("recv : idx: {}, j: {} done", i, j);
                            *rrequest = None;
                        }
                    }
                    if srequest.is_some() || rrequest.is_some() {
                        all_done = false
                    }
                }
                if all_done {
                    finished += 1;
                    *req = None;
                }
            }
        }

        if finished == args.try_count {
            break;
        }
    }

    // stop timer
    let elapsed = start.elapsed();
    print_stat(&args, &elapsed);
}

pub(crate) fn client(args: Args) {
    let (streams, comms): (Vec<TcpStream>, Vec<Vec<(Comm, Comm)>>) = args
        .address
        .split(',')
        .map(|addr| {
            info!("connecting to {}", addr);
            let mut stream = loop {
                let res = TcpStream::connect(&addr);
                if res.is_ok() {
                    break res.unwrap();
                }
                // sleep 1s
                std::thread::sleep(std::time::Duration::from_secs(1));
            };

            let comms = (0..args.nchannel)
                .map(|_| {
                    let mut buffer = [0u8; 4];
                    stream.read(buffer.as_mut()).unwrap();
                    let size = u32::from_le_bytes(buffer);
                    let mut handle = vec![0u8; size as usize];
                    stream.read(handle.as_mut()).unwrap();
                    info!("received handle: {:?}", handle);

                    let (lcomm, lhandle) = nccl_net::listen().unwrap();

                    // send size of handle
                    let size = lhandle.len() as u32;
                    stream.write_all(&size.to_le_bytes()).unwrap();
                    // send handle
                    stream.write_all(&lhandle).unwrap();

                    let mut scomm: Option<Comm> = None;
                    let mut rcomm: Option<Comm> = None;

                    loop {
                        if scomm.is_none() {
                            scomm = nccl_net::connect(handle.as_slice()).unwrap();
                        }
                        if rcomm.is_none() {
                            rcomm = nccl_net::accept(&lcomm).unwrap();
                        }
                        if scomm.is_some() && rcomm.is_some() {
                            break;
                        }
                    }

                    let scomm = scomm.unwrap();
                    let rcomm = rcomm.unwrap();
                    (scomm, rcomm)
                })
                .collect::<Vec<_>>();
            (stream, comms) // return stream to keep the socket open until we finish
        })
        .unzip();

    let comms = transpose(comms);

    info!("client connected");

    let args = Arc::new(args);

    let hs = comms
        .into_iter()
        .map(|comm| {
            let args = Arc::clone(&args);
            std::thread::spawn(move || {
                if args.data_type == DataType::F32 {
                    do_client::<f32>(args.as_ref(), comm);
                } else if args.data_type == DataType::F16 {
                    do_client::<f16>(args.as_ref(), comm);
                }
            })
        })
        .collect::<Vec<_>>();
    hs.into_iter().for_each(|h| h.join().unwrap());
    drop(streams);
}

pub(crate) fn bench(args: Args) {
    let listener =
        TcpListener::bind(format!("{}:{}", args.address, args.port)).expect("failed to bind");

    let (streams, comms): (Vec<TcpStream>, Vec<Vec<(Comm, Comm)>>) = (0..args.nrank)
        .map(|_| {
            let (mut stream, _) = listener.accept().unwrap();
            let comms = (0..args.nchannel)
                .map(|_| {
                    let (lcomm, handle) = nccl_net::listen().unwrap();
                    // send size of handle
                    let size = handle.len() as u32;
                    stream.write_all(&size.to_le_bytes()).unwrap();
                    // send handle
                    stream.write_all(&handle).unwrap();

                    let mut buffer = [0u8; 4];
                    stream.read(buffer.as_mut()).unwrap();
                    let size = u32::from_le_bytes(buffer);
                    let mut handle = vec![0u8; size as usize];
                    stream.read(handle.as_mut()).unwrap();
                    info!("received handle: {:?}", handle);

                    let mut scomm: Option<Comm> = None;
                    let mut rcomm: Option<Comm> = None;

                    loop {
                        if scomm.is_none() {
                            scomm = nccl_net::connect(handle.as_slice()).unwrap();
                        }
                        if rcomm.is_none() {
                            rcomm = nccl_net::accept(&lcomm).unwrap();
                        }
                        if scomm.is_some() && rcomm.is_some() {
                            break;
                        }
                    }

                    let scomm = scomm.unwrap();
                    let rcomm = rcomm.unwrap();
                    (scomm, rcomm)
                })
                .collect::<Vec<_>>();
            (stream, comms) // return stream to keep the socket open until we finish
        })
        .unzip();

    let comms = transpose(comms);

    info!("bench connected");

    let args = Arc::new(args);

    let hs = comms
        .into_iter()
        .map(|comm| {
            let args = Arc::clone(&args);
            std::thread::spawn(move || {
                if args.data_type == DataType::F32 {
                    do_client::<f32>(args.as_ref(), comm);
                } else if args.data_type == DataType::F16 {
                    do_client::<f16>(args.as_ref(), comm);
                }
            })
        })
        .collect::<Vec<_>>();
    hs.into_iter().for_each(|h| h.join().unwrap());
    drop(streams);
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::initialize;
    use clap::Parser;

    #[test]
    fn test_bench() {
        initialize();
        let b = std::thread::spawn(|| {
            let count = format!("{}", 1024 * 1024);
            let args = Args::parse_from([
                "--bench",
                "--address",
                "127.0.0.1",
                "--port",
                "8080",
                "--count",
                &count,
            ]);
            bench(args);
        });
        let c = std::thread::spawn(|| {
            let count = format!("{}", 1024 * 1024);
            let args =
                Args::parse_from(["--client", "--address", "127.0.0.1:8080", "--count", &count]);
            client(args);
        });
        b.join().unwrap();
        c.join().unwrap();
    }
}
