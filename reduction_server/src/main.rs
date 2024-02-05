/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

#![feature(c_variadic)]
#![feature(portable_simd)]
#![feature(min_specialization)]

use std::collections::HashMap;
use std::fmt::Debug;
use std::hint;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use aligned_box::AlignedBox;
use clap::{Parser, ValueEnum};
use half::f16;
use half::slice::HalfFloatSliceExt;
use log::{info, trace, warn};
use num_traits::FromPrimitive;

mod nccl_net;
use nccl_net::{Comm, Request};

mod partitioned_vec;
use partitioned_vec::PartitionedVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DataType {
    F32,
    F16,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    verbose: bool,

    #[arg(short, long)]
    client: bool,

    #[arg(short, long, default_value = "8918")]
    port: u16,

    #[arg(short, long, default_value = "0.0.0.0")]
    address: String,

    #[arg(long, default_value = "1024")]
    count: usize,

    #[arg(long, default_value = "100")]
    try_count: usize,

    #[arg(long, default_value = "2", help = "threads per reduce job")]
    reduce_threads: usize,

    #[arg(long, default_value = "2")]
    reduce_jobs: usize,

    #[arg(long, default_value = "0")] // 0: = nrank
    recv_threads: usize,

    #[arg(long, default_value = "0")] // 0: = nrank
    send_threads: usize,

    #[arg(long, default_value = "1")]
    nrank: usize,

    #[arg(long, default_value = "f32")]
    data_type: DataType,
}

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

struct WorkingMemory {
    recv_bufs: Vec<AlignedBox<[f32]>>,
    send_buf: AlignedBox<[f32]>,
}

impl WorkingMemory {
    fn new(count: usize, num_recv: usize) -> Self {
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

trait Reduce<T> {
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
        .map(|_| WorkingMemory::new(args.count, args.recv_threads))
        .collect::<Vec<_>>();

    loop {
        for (job_idx, (send_ready, recv_ready, send_buf, recv_bufs)) in jobs.iter_mut().enumerate()
        {
            trace!("rank({})/job({}) reduce wait recv", i, job_idx);

            loop {
                hint::spin_loop();
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

    loop {
        if rank.load(std::sync::atomic::Ordering::Relaxed) == args.nrank {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    info!("send thread({}) all ranks get connected!", i);

    for (idx, (readys, send)) in sends.iter().enumerate().cycle() {
        for ready in readys.iter() {
            loop {
                hint::spin_loop();
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
            hint::spin_loop();
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
            hint::spin_loop();
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
            "rank({})/job({}) send latency: {}us",
            i,
            idx,
            start.elapsed().as_micros()
        );
    }
}

fn vec_of_none<T>(n: usize) -> Vec<Option<T>> {
    std::iter::repeat_with(|| None).take(n).collect()
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

    loop {
        if rank.load(std::sync::atomic::Ordering::Relaxed) == args.nrank {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    info!("recv thread({}) all ranks get connected!", i);

    loop {
        for (job_idx, (readys, recv)) in recvs.iter_mut().enumerate() {
            for ready in readys.iter() {
                loop {
                    hint::spin_loop();
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
                hint::spin_loop();
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
                hint::spin_loop();
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
                "rank({})/job({}) recv latency: {}us",
                i,
                job_idx,
                start.elapsed().as_micros()
            );
        }
    }
}

fn do_server<T: Float + 'static>(args: Args) {
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

    // tranpose readys[job][thread]
    let (send_readys, recv_readys): (Vec<_>, Vec<_>) = (0..args.reduce_jobs)
        .map(|i| {
            (0..args.reduce_threads)
                .map(|j| Option::take(&mut readys[j][i]).unwrap())
                .unzip::<_, _, Vec<_>, Vec<_>>()
        })
        .unzip();

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

    let hs = (0..args.nrank).map(|_| {
        let (socket, _) = listener.accept().unwrap();
        let idx = rank.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rcomm_ch = recv_chs[idx % recv_chs.len()].clone();
        let scomm_ch = send_chs[idx % send_chs.len()].clone();
        let rank = Arc::clone(&rank);
        std::thread::spawn(move || handle_connection(socket, idx, &rank, rcomm_ch, scomm_ch))
    }).collect::<Vec<_>>();
    hs.into_iter().for_each(|h| h.join().unwrap());
}

fn server(args: Args) {
    if args.data_type == DataType::F32 {
        do_server::<f32>(args);
    } else if args.data_type == DataType::F16 {
        do_server::<f16>(args);
    }
}

trait Float: num_traits::Float + FromPrimitive + Default + Sync + Send + std::fmt::Debug {}

impl Float for f32 {}
impl Float for f16 {}

fn do_client<T: Float>(args: Args, scomm: Comm, rcomm: Comm) {
    let size = args.count * std::mem::size_of::<T>();
    let initial: T = T::from_f32(2.0).unwrap();
    let mut sbuf =
        AlignedBox::<[T]>::slice_from_value(alignment(size), args.count, initial).unwrap();
    let mut rbuf = AlignedBox::<[T]>::slice_from_default(alignment(size), args.count).unwrap();

    let s_mhandle = nccl_net::reg_mr(&scomm, &mut sbuf).unwrap();
    let r_mhandle = nccl_net::reg_mr(&rcomm, &mut rbuf).unwrap();

    let tag = 0x69;

    for i in 0..args.try_count {
        let mut srequest: Option<Request> = None;
        let mut rrequest: Option<Request> = None;

        loop {
            if srequest.is_none() {
                srequest = nccl_net::isend(&scomm, &s_mhandle, &sbuf, tag).unwrap();
            }
            if rrequest.is_none() {
                rrequest = nccl_net::irecv(&rcomm, &r_mhandle, &mut rbuf, tag).unwrap();
            }
            if srequest.is_some() && rrequest.is_some() {
                break;
            }
        }

        let mut recv_done = false;
        let mut send_done = false;

        loop {
            if !send_done {
                (send_done, _) = nccl_net::test(srequest.as_ref().unwrap()).unwrap();
            }
            if !recv_done {
                (recv_done, _) = nccl_net::test(rrequest.as_ref().unwrap()).unwrap();
            }
            if send_done && recv_done {
                break;
            }
        }

        let check = rbuf.iter().all(|v| {
            (*v - initial * T::from_f32(args.nrank as f32).unwrap()).abs()
                < T::from_f32(1e-6).unwrap()
        });

        info!("check({}): {}", i, check);
        if !check {
            info!("{:?}", rbuf.as_ref());
        }
    }
}

fn client(args: Args) {
    let mut stream = TcpStream::connect(format!("{}:{}", args.address, args.port))
        .expect("Could not connect to server");
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

    info!("client connected");

    if args.data_type == DataType::F32 {
        do_client::<f32>(args, scomm, rcomm);
    } else if args.data_type == DataType::F16 {
        do_client::<f16>(args, scomm, rcomm);
    }
}

fn alignment(size: usize) -> usize {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    (size + page - 1) & !(page - 1)
}

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
    } else {
        server(args);
        return;
    }
}