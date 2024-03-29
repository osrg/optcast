/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::collections::HashMap;
use std::hint;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use half::{bf16, f16};
use log::{error, info, trace, warn};

use crate::reduce::{Reduce, WorkingMemory};
use crate::utils::*;

use crate::nccl_net;
use crate::nccl_net::{Comm, Request};

use crate::partitioned_vec::PartitionedVec;

fn handle_connection(
    stream: std::net::TcpStream,
    idx: usize,
    rank: &AtomicUsize,
    rcomm_ch: std::sync::mpsc::Sender<(usize, Comm)>,
    scomm_ch: std::sync::mpsc::Sender<(usize, Comm)>,
) {
    let (lcomm, handle) = nccl_net::listen().unwrap();

    let mut stream = stream;

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

    info!("server connected");
    rcomm_ch.send((idx, rcomm.unwrap())).unwrap();
    scomm_ch.send((idx, scomm.unwrap())).unwrap();

    let ret = stream.read(buffer.as_mut());

    info!("handle_connection: exiting ret {:?}", ret);

    rank.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
}

fn reduce_loop<T: Float>(
    i: usize,
    args: &Args,
    rank: &AtomicUsize,
    mut jobs: Vec<(
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<PartitionedVec<T>>,
        Vec<Arc<PartitionedVec<T>>>,
    )>,
) {
    info!("reduce thread({})", i);

    loop {
        if rank.load(std::sync::atomic::Ordering::Relaxed) == args.nrank {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    info!("reduce thread({}) all ranks get connected!", i);

    let mut mems = (0..jobs.len())
        .map(|_| WorkingMemory::new(args.count / args.reduce_threads, args.recv_threads))
        .collect::<Vec<_>>();

    loop {
        for (job_idx, (send_ready, recv_ready, send_buf, recv_bufs)) in jobs.iter_mut().enumerate()
        {
            trace!("rank({})/job({}) reduce wait recv", i, job_idx);

            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }
                let send_ready = send_ready.load(std::sync::atomic::Ordering::Relaxed);
                let send_expect = (1 << args.send_threads) - 1;
                let recv_ready = recv_ready.load(std::sync::atomic::Ordering::Relaxed);
                let recv_expect = (1 << args.recv_threads) - 1;
                //            trace!(
                //                "[reduce] job({})/({}) recv ready: 0b{:016b}, expect: 0b{:016b}",
                //                job_idx,
                //                offset,
                //                ready,
                //                expect
                //            );
                if send_ready == send_expect && recv_ready == recv_expect {
                    break;
                }
                if rank.load(std::sync::atomic::Ordering::Relaxed) != args.nrank {
                    warn!("rank != nrank");
                    warn!("reduce thread({}) exit.", i);
                    return;
                }
            }

            trace!("rank({})/job({}) reduce start", i, job_idx);
            // start timer for performance measurement
            let start = std::time::Instant::now();
            {
                let mut send_buf = send_buf.parts[i].lock().unwrap();
                let recv_buf_guards = recv_bufs
                    .iter()
                    .map(|v| v.parts[i].lock().unwrap())
                    .collect::<Vec<_>>();
                let recv_bufs = recv_buf_guards
                    .iter()
                    .map(|v| v.as_ref())
                    .collect::<Vec<_>>();
                send_buf
                    .reduce(&recv_bufs, Some(&mut mems[job_idx]))
                    .unwrap();
            }
            // stop timer
            let elapsed = start.elapsed();
            trace!(
                "rank({})/job({}) reduce latency: {}us",
                i,
                job_idx,
                elapsed.as_micros()
            );

            recv_ready.store(0, std::sync::atomic::Ordering::Relaxed);
            send_ready.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn send_loop<T: Float>(
    i: usize,
    args: &Args,
    rank: &AtomicUsize,
    sends: Vec<(Vec<Arc<AtomicUsize>>, Arc<PartitionedVec<T>>)>,
    rx: std::sync::mpsc::Receiver<(usize, Comm)>,
) {
    let nrank = args.nrank;
    let nsends = args.send_threads;
    info!(
        "send thread({}) sends.len(): {} waiting all ranks get connected.",
        i,
        sends.len(),
    );

    let comms = (0..nrank / nsends)
        .map(|_| rx.recv().unwrap())
        .collect::<Vec<_>>();
    let sends = sends
        .iter()
        .map(|v| {
            (
                &v.0,
                comms
                    .iter()
                    .map(|(_, comm)| {
                        let mh = nccl_net::reg_mr(comm, &v.1.lock()).unwrap();
                        (comm, mh, &v.1)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let size = args.count * {
        if args.data_type == DataType::F32 {
            4
        } else {
            2
        }
    } as usize;

    loop {
        if rank.load(std::sync::atomic::Ordering::Relaxed) == args.nrank {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    info!(
        "send thread({}) all ranks get connected!, size: {}",
        i, size
    );

    for (idx, (readys, send)) in sends.iter().enumerate().cycle() {
        for ready in readys.iter() {
            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }
                let ready = ready.load(std::sync::atomic::Ordering::Relaxed);
                //                trace!(
                //                    "[send] rank({})/job({}) send ready: 0b{:016b}",
                //                    i,
                //                    idx,
                //                    ready,
                //                );
                if ready & (1 << i) == 0 {
                    break;
                }
                if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                    warn!("rank != nrank");
                    warn!("send thread({}) exit.", i);
                    return;
                }
            }
        }
        trace!("rank({})/job({}) send start", i, idx);

        let mut reqs = vec_of_none(send.len());
        loop {
            if cfg!(no_spinloop) {
                std::thread::sleep(NO_SPINLOOP_INTERVAL);
            } else {
                hint::spin_loop();
            }
            if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                warn!("rank != nrank");
                warn!("send thread({}) exit.", i);
                return;
            }

            let mut done = true;
            for (j, (comm, mh, buf)) in send.iter().enumerate() {
                if reqs[j].is_none() {
                    reqs[j] = nccl_net::isend(comm, mh, &buf.lock(), 0x69).unwrap();
                    if reqs[j].is_none() {
                        done = false;
                    }
                }
            }

            if done {
                break;
            }
        }
        trace!("rank({})/job({}) send requested", i, idx);
        let start = std::time::Instant::now();

        loop {
            if cfg!(no_spinloop) {
                std::thread::sleep(NO_SPINLOOP_INTERVAL);
            } else {
                hint::spin_loop();
            }

            if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                warn!("rank != nrank");
                warn!("send thread({}) exit.", i);
                return;
            }

            let mut done = true;
            for (j, _) in send.iter().enumerate() {
                if reqs[j].is_some() {
                    let (d, _) = nccl_net::test(reqs[j].as_ref().unwrap()).unwrap();
                    if d {
                        reqs[j] = None;
                    } else {
                        done = false;
                    }
                }
            }
            if done {
                break;
            }
        }

        for ready in readys.iter() {
            ready.fetch_add(1 << i, std::sync::atomic::Ordering::Relaxed);
        }

        trace!(
            "rank({})/job({}) send latency: {}us, {:.2}Gbps",
            i,
            idx,
            start.elapsed().as_micros(),
            (size * 8) as f64 / start.elapsed().as_secs_f64() * 1e-9
        );
    }
}

fn recv_loop<T: Float>(
    i: usize,
    args: &Args,
    rank: &AtomicUsize,
    mut recvs: Vec<(
        Vec<Arc<AtomicUsize>>,
        Vec<(usize, Option<Arc<PartitionedVec<T>>>)>,
    )>, // len = reduce-threads
    rx: std::sync::mpsc::Receiver<(usize, Comm)>,
) {
    let nrank = args.nrank;
    let nrecvs = args.recv_threads;
    info!(
        "recv thread: {}, recvs: {:?}",
        i,
        recvs
            .iter()
            .map(|v| v.1.iter().map(|(j, _)| j).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );

    let comms: HashMap<usize, Comm> = (0..nrank / nrecvs)
        .map(|_| rx.recv().unwrap())
        .collect::<HashMap<_, _>>();
    let mut recvs = recvs
        .iter_mut()
        .map(|v| {
            (
                &v.0,
                v.1.iter_mut()
                    .map(|(idx, buf)| {
                        let comm = comms.get(idx).unwrap();
                        let mh = nccl_net::reg_mr(comm, &buf.as_ref().unwrap().lock()).unwrap();
                        (comm, mh, Option::take(buf).unwrap())
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();

    let size = args.count * {
        if args.data_type == DataType::F32 {
            4
        } else {
            2
        }
    } as usize;

    loop {
        if rank.load(std::sync::atomic::Ordering::Relaxed) == args.nrank {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    info!(
        "recv thread({}) all ranks get connected!, size: {}",
        i, size
    );

    loop {
        for (job_idx, (readys, recv)) in recvs.iter_mut().enumerate() {
            for ready in readys.iter() {
                loop {
                    if cfg!(no_spinloop) {
                        std::thread::sleep(NO_SPINLOOP_INTERVAL);
                    } else {
                        hint::spin_loop();
                    }
                    let ready = ready.load(std::sync::atomic::Ordering::Relaxed);
                    //                    trace!(
                    //                        "[recv] rank({})/job({}) recv ready: 0b{:016b}",
                    //                        i,
                    //                        job_idx,
                    //                        ready,
                    //                    );
                    if ready & (1 << i) == 0 {
                        break;
                    }
                    if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                        warn!("rank != nrank");
                        warn!("recv thread({}) exit.", i);
                        return;
                    }
                }
            }
            trace!("rank({})/job({}) recv start", i, job_idx);

            let mut reqs = vec_of_none(recv.len());
            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }
                if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                    warn!("rank != nrank");
                    warn!("recv thread({}) exit.", i);
                    return;
                }

                let mut done = true;
                for (j, (comm, mh, buf)) in recv.iter_mut().enumerate() {
                    if reqs[j].is_none() {
                        reqs[j] = nccl_net::irecv(comm, mh, &mut buf.lock(), 0x69).unwrap();
                        if reqs[j].is_none() {
                            done = false;
                        }
                    }
                }

                if done {
                    break;
                }
            }

            trace!("rank({})/job({}) recv requested", i, job_idx);
            let start = std::time::Instant::now();

            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }

                if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                    warn!("rank != nrank");
                    warn!("recv thread({}) exit.", i);
                    return;
                }

                let mut done = true;
                for (j, _) in recv.iter().enumerate() {
                    if reqs[j].is_some() {
                        let (d, _) = nccl_net::test(reqs[j].as_ref().unwrap()).unwrap();
                        if d {
                            reqs[j] = None;
                        } else {
                            done = false;
                        }
                    }
                }
                if done {
                    break;
                }
            }

            for ready in readys.iter() {
                ready.fetch_add(1 << i, std::sync::atomic::Ordering::Relaxed);
            }

            trace!(
                "rank({})/job({}) recv latency: {}us, {:.2}Gbps",
                i,
                job_idx,
                start.elapsed().as_micros(),
                (size * 8) as f64 / start.elapsed().as_secs_f64() * 1e-9
            );
        }
    }
}

fn upstream_loop<T: Float>(
    args: &Args,
    rank: &AtomicUsize,
    mut jobs: Vec<(
        Arc<AtomicUsize>,
        Vec<Arc<AtomicUsize>>,
        Arc<PartitionedVec<T>>,
    )>,
) {
    let nrank = args.nrank;

    info!("connecting to upstream {}", args.upstream);
    let mut stream = loop {
        let res = TcpStream::connect(&args.upstream);
        if res.is_ok() {
            break res.unwrap();
        }
        // sleep 1s
        std::thread::sleep(std::time::Duration::from_secs(1));
    };

    let mut buffer = [0u8; 4];
    stream.read(buffer.as_mut()).unwrap();
    let size = u32::from_le_bytes(buffer);
    let mut handle = vec![0u8; size as usize];
    stream.read(handle.as_mut()).unwrap();

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

    let mhs = jobs
        .iter()
        .map(|(_, _, buf)| {
            let send_mh = nccl_net::reg_mr(&scomm, &buf.lock()).unwrap();
            let recv_mh = nccl_net::reg_mr(&rcomm, &buf.lock()).unwrap();
            (send_mh, recv_mh)
        })
        .collect::<Vec<_>>();

    loop {
        if rank.load(std::sync::atomic::Ordering::Relaxed) == args.nrank {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    info!("upstream connected");
    let tag = 0x69;

    loop {
        for (idx, (send_ready, reduce_readys, buf)) in jobs.iter_mut().enumerate() {
            for reduce_ready in reduce_readys.iter() {
                loop {
                    if cfg!(no_spinloop) {
                        std::thread::sleep(NO_SPINLOOP_INTERVAL);
                    } else {
                        hint::spin_loop();
                    }

                    let reduce_ready = reduce_ready.load(std::sync::atomic::Ordering::Relaxed);
                    if reduce_ready == 0 {
                        break;
                    }
                    if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                        warn!("rank != nrank");
                        warn!("upstream thread({}) exit.", 0);
                        return;
                    }
                }
            }

            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }
                let send_ready = send_ready.load(std::sync::atomic::Ordering::Relaxed);
                let send_expect = (1 << args.send_threads) - 1;
                if send_ready == send_expect {
                    break;
                }
                if rank.load(std::sync::atomic::Ordering::Relaxed) != nrank {
                    warn!("rank != nrank");
                    warn!("upstream thread({}) exit.", 0);
                    return;
                }
            }

            let (send_mh, recv_mh) = &mhs[idx];
            let mut srequest: Option<Request> = None;
            let mut rrequest: Option<Request> = None;

            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }

                if srequest.is_none() {
                    srequest = nccl_net::isend(&scomm, send_mh, buf.lock().as_ref(), tag).unwrap();
                    if srequest.is_some() {
                        trace!("upstream send  : idx: {} start", idx);
                    }
                }
                if rrequest.is_none() {
                    rrequest = nccl_net::irecv(&rcomm, recv_mh, buf.lock().as_mut(), tag).unwrap();
                    if srequest.is_some() {
                        trace!("upstream recv : idx: {} start", idx);
                    }
                }
                if srequest.is_some() && rrequest.is_some() {
                    break;
                }
            }

            loop {
                if cfg!(no_spinloop) {
                    std::thread::sleep(NO_SPINLOOP_INTERVAL);
                } else {
                    hint::spin_loop();
                }

                if srequest.is_some() {
                    match nccl_net::test(&srequest.as_ref().unwrap()) {
                        Ok((send_done, _)) => {
                            if send_done {
                                trace!("upstream send  : idx: {} done", idx);
                                srequest = None;
                            }
                        }
                        Err(e) => {
                            error!("upstream send  : idx: {} error: {:?}", idx, e);
                            return;
                        }
                    }
                }
                if rrequest.is_some() {
                    match nccl_net::test(&rrequest.as_ref().unwrap()) {
                        Ok((recv_done, _)) => {
                            if recv_done {
                                trace!("upstream recv  : idx: {} done", idx);
                                rrequest = None;
                            }
                        }
                        Err(e) => {
                            error!("upstream recv : idx: {} error: {:?}", idx, e);
                            return;
                        }
                    }
                }
                if srequest.is_none() && rrequest.is_none() {
                    break;
                }
            }

            for reduce_ready in reduce_readys.iter_mut() {
                reduce_ready.store(
                    (1 << args.send_threads) - 1,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

            send_ready.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn do_server<T: Float + 'static>(args: Args) {
    let mut args = args;

    if args.recv_threads == 0 {
        args.recv_threads = args.nrank
    }

    if args.send_threads == 0 {
        args.send_threads = args.nrank
    }

    let listener =
        TcpListener::bind(format!("{}:{}", args.address, args.port)).expect("failed to bind");

    let rank = Arc::new(AtomicUsize::new(0));
    let size = args.count * std::mem::size_of::<T>();

    let args = Arc::new(args);

    // memory allocation
    let bufs = (0..args.reduce_jobs)
        .map(|_| {
            let sbuf = Arc::new(
                PartitionedVec::<T>::new(alignment(size), args.count, args.reduce_threads).unwrap(),
            );

            let rbufs = (0..args.nrank)
                .map(|_| {
                    Arc::new(
                        PartitionedVec::new(alignment(size), args.count, args.reduce_threads)
                            .unwrap(),
                    )
                })
                .collect::<Vec<_>>();

            (sbuf, rbufs)
        })
        .collect::<Vec<_>>();

    // launch reduce threads
    let mut readys = (0..args.reduce_threads)
        .map(|i| {
            let rank = Arc::clone(&rank);
            let jobs = bufs
                .iter()
                .map(|(sbuf, rbufs)| {
                    let send_ready = Arc::new(AtomicUsize::new((1 << args.send_threads) - 1));
                    let recv_ready = Arc::new(AtomicUsize::new(0));

                    let recv_bufs = rbufs
                        .iter()
                        .map(|rbuf| Arc::clone(rbuf))
                        .collect::<Vec<_>>();
                    (send_ready, recv_ready, Arc::clone(sbuf), recv_bufs)
                })
                .collect::<Vec<_>>();

            let readys = jobs
                .iter()
                .map(|v| Some((Arc::clone(&v.0), Arc::clone(&v.1))))
                .collect::<Vec<_>>();

            let args = Arc::clone(&args);
            std::thread::spawn(move || reduce_loop(i, &args, &rank, jobs));
            readys
        })
        .collect::<Vec<_>>();

    // transpose readys[reduce_threads][reduce_jobs] to readys[reduce_jobs][reduce_threads]
    let (send_readys, recv_readys): (Vec<_>, Vec<_>) = (0..args.reduce_jobs)
        .map(|i| {
            (0..args.reduce_threads)
                .map(|j| Option::take(&mut readys[j][i]).unwrap())
                .unzip::<_, _, Vec<_>, Vec<_>>()
        })
        .unzip();

    let send_readys = if args.upstream.is_empty() {
        send_readys
    } else {
        // launch upstream thread
        let args = Arc::clone(&args);
        let jobs = send_readys
            .into_iter()
            .enumerate()
            .map(|(i, send_ready)| {
                let ready = Arc::new(AtomicUsize::new((1 << args.send_threads) - 1));
                let (sbuf, _) = &bufs[i];
                (ready, send_ready, Arc::clone(sbuf))
            })
            .collect::<Vec<_>>();

        let readys = jobs
            .iter()
            .map(|(ready, _, _)| vec![Arc::clone(ready)])
            .collect::<Vec<_>>();

        let rank = Arc::clone(&rank);
        std::thread::spawn(move || upstream_loop(&args, &rank, jobs));
        readys
    };

    // launch send threads
    let send_chs = (0..args.send_threads)
        .map(|send_idx| {
            let rank = Arc::clone(&rank);
            let sends = bufs
                .iter()
                .zip(&send_readys)
                .map(|((sbuf, _), readys)| {
                    (
                        readys.iter().map(|v| Arc::clone(v)).collect::<Vec<_>>(),
                        Arc::clone(sbuf),
                    )
                })
                .collect::<Vec<_>>();

            let (tx, rx) = std::sync::mpsc::channel();
            let args = Arc::clone(&args);
            std::thread::spawn(move || send_loop(send_idx, &args, &rank, sends, rx));
            tx
        })
        .collect::<Vec<_>>();

    // launch recv threads
    let recv_chs = (0..args.recv_threads)
        .map(|recv_idx| {
            let rank = Arc::clone(&rank);
            let recvs = bufs
                .iter()
                .zip(&recv_readys)
                .map(|((_, rbufs), readys)| {
                    (
                        readys.iter().map(|v| Arc::clone(v)).collect::<Vec<_>>(),
                        rbufs
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| j % args.recv_threads == recv_idx)
                            .map(|(k, rbuf)| (k, Some(Arc::clone(rbuf))))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            let (tx, rx) = std::sync::mpsc::channel();
            let args = Arc::clone(&args);
            std::thread::spawn(move || recv_loop(recv_idx, &args, &rank, recvs, rx));
            tx
        })
        .collect::<Vec<_>>();

    let hs = (0..args.nrank)
        .map(|_| {
            let (socket, _) = listener.accept().unwrap();
            let idx = rank.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let rcomm_ch = recv_chs[idx % recv_chs.len()].clone();
            let scomm_ch = send_chs[idx % send_chs.len()].clone();
            let rank = Arc::clone(&rank);
            std::thread::spawn(move || handle_connection(socket, idx, &rank, rcomm_ch, scomm_ch))
        })
        .collect::<Vec<_>>();
    hs.into_iter().for_each(|h| h.join().unwrap());
}

pub(crate) fn server(args: Args) {
    if args.data_type == DataType::F32 {
        do_server::<f32>(args);
    } else if args.data_type == DataType::F16 {
        do_server::<f16>(args);
    } else if args.data_type == DataType::BF16 {
        do_server::<bf16>(args);
    }
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::client;
    use crate::utils::tests::initialize;
    use clap::Parser;

    fn do_test(dt: &str) {
        initialize();
        let nrank = 4;
        let server = {
            let dt = dt.to_string();
            std::thread::spawn(move || {
                let nrank = format!("{}", nrank);
                let args = Args::parse_from([
                    "--verbose", // doesn't work without specifying a flag that doesn't take an argument
                    "--port",
                    "8080",
                    "--data-type",
                    &dt,
                    "--nrank",
                    &nrank,
                    "--nreq",
                    "1", // when using socket plugin, concurrent recv/send requests doesn't work
                ]);
                server(args);
            })
        };
        (0..nrank)
            .map(|_| {
                let dt = dt.to_string();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let args = Args::parse_from([
                        "--client",
                        "--address",
                        "127.0.0.1:8080",
                        "--data-type",
                        &dt,
                        "--nreq",
                        "1", // when using socket plugin, concurrent recv/send requests doesn't work
                    ]);
                    client(args);
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());
        server.join().unwrap();
    }

    fn do_test_upstream(dt: &str) {
        initialize();
        let nrank = 2;
        let root = {
            let dt = dt.to_string();
            std::thread::spawn(move || {
                let nrank = format!("{}", nrank);
                let args = Args::parse_from([
                    "--verbose", // doesn't work without specifying a flag that doesn't take an argument
                    "--port",
                    "8080",
                    "--data-type",
                    &dt,
                    "--nrank",
                    &nrank,
                ]);
                server(args);
            })
        };
        (0..nrank)
            .map(|i| {
                let port = format!("{}", 8081 + i);
                let parent = {
                    let dt = dt.to_string();
                    let port = port.to_string();
                    std::thread::spawn(move || {
                        let nrank = format!("{}", nrank);
                        let args = Args::parse_from([
                            "--verbose", // doesn't work without specifying a flag that doesn't take an argument
                            "--upstream",
                            "localhost:8080",
                            "--port",
                            &port,
                            "--data-type",
                            &dt,
                            "--nrank",
                            &nrank,
                        ]);
                        server(args);
                    })
                };
                let children = (0..nrank).map(move |_| {
                    let dt = dt.to_string();
                    let port = port.to_string();
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        let address = format!("127.0.0.1:{}", port);
                        let args = Args::parse_from([
                            "--client",
                            "--address",
                            &address,
                            "--data-type",
                            &dt,
                            "--nreq",
                            "1", // when using socket plugin, concurrent recv/send requests doesn't work
                        ]);
                        client(args);
                    })
                });
                vec![parent].into_iter().chain(children)
            })
            .flatten()
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());

        root.join().unwrap();
    }

    #[test]
    fn test_server_with_upstream_f32() {
        do_test_upstream("f32");
    }

    #[test]
    fn test_server_f32() {
        do_test("f32");
    }

    #[test]
    fn test_server_f16() {
        do_test("f16");
    }

    #[test]
    fn test_server_bf16() {
        do_test("bf16");
    }
}
