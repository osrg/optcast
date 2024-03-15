/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::hint;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use aligned_box::AlignedBox;
use half::{bf16, f16};
use log::{info, trace};

use crate::reduce::{Reduce, WorkingMemory};
use crate::utils::*;

use crate::nccl_net;
use crate::nccl_net::Comm;

struct Task<'a, T> {
    task_id: usize,
    tasks: Vec<(
        usize,
        Arc<AtomicUsize>,
        Vec<nccl_net::MemoryHandle<'a>>,
        Vec<Arc<Mutex<AlignedBox<[T]>>>>,
    )>,
    comms: &'a Vec<Comm>,
    task_ready: Arc<AtomicUsize>,
    args: &'a Args,
    req: Option<Vec<Option<nccl_net::Request>>>,
    next: usize,
    count: usize,
    reqcount: usize,
    try_count: usize,
    timer: std::time::Instant,
}

impl<'a, T> Task<'a, T> {
    fn new(
        task_id: usize,
        comms: &'a Vec<Comm>,
        tasks: Vec<(usize, Arc<AtomicUsize>, Vec<Arc<Mutex<AlignedBox<[T]>>>>)>,
        args: &'a Args,
        task_ready: Arc<AtomicUsize>,
    ) -> Self {
        let tasks = tasks
            .into_iter()
            .map(|(idx, ready, bufs)| {
                let mhs = comms
                    .iter()
                    .enumerate()
                    .map(|(i, comm)| nccl_net::reg_mr(comm, &bufs[i].lock().unwrap()).unwrap())
                    .collect::<Vec<_>>();
                (idx, ready, mhs, bufs)
            })
            .collect::<Vec<_>>();
        let nring = args.address.split(",").count();
        let tasks_len = tasks.len();
        Task {
            task_id,
            tasks,
            comms,
            task_ready,
            args,
            req: None,
            next: 0,
            count: 0,
            reqcount: 0,
            try_count: args.try_count * tasks_len / args.nreq / nring,
            timer: std::time::Instant::now(),
        }
    }

    fn recv(
        &self,
        comm: &Comm,
        mh: &nccl_net::MemoryHandle<'a>,
        buf: &Mutex<AlignedBox<[T]>>,
    ) -> Option<nccl_net::Request> {
        let mut req = None;
        while req.is_none() {
            req = nccl_net::irecv(comm, mh, &mut buf.lock().unwrap(), 0x69).unwrap();
        }
        req
    }

    fn send(
        &self,
        comm: &Comm,
        mh: &nccl_net::MemoryHandle<'a>,
        buf: &Mutex<AlignedBox<[T]>>,
    ) -> Option<nccl_net::Request> {
        let mut req = None;
        while req.is_none() {
            req = nccl_net::isend(comm, mh, &buf.lock().unwrap(), 0x69).unwrap();
        }
        req
    }

    fn progress(&mut self, is_recv: bool) -> bool {
        let task = &self.tasks[self.next];
        let idx = self.next;
        let (i, buf_ready, mhs, bufs) = task;

        if self.count == self.try_count {
            return false; // noop
        }

        let op = if is_recv { Self::recv } else { Self::send };
        let (opname, ready_value, done_value) = if is_recv {
            ("recv", 0, self.args.reduce_threads)
        } else {
            ("send", self.args.reduce_threads, 0)
        };

        if self.reqcount < self.try_count
            && self.req.is_none()
            && buf_ready.load(std::sync::atomic::Ordering::Relaxed) == ready_value
            && self.task_ready.load(std::sync::atomic::Ordering::Relaxed) == self.task_id
        {
            self.req = Some(
                self.comms
                    .iter()
                    .enumerate()
                    .map(|(j, comm)| {
                        let req = op(self, comm, &mhs[j], &bufs[j]);
                        trace!(
                            "{}  : task_id: {}, idx: {}, i: {}, j: {}, start",
                            opname,
                            self.task_id,
                            idx,
                            i,
                            j
                        );
                        req
                    })
                    .collect::<Vec<_>>(),
            );
            if self.reqcount == 0 {
                self.timer = std::time::Instant::now();
            }
            self.reqcount += 1;
            self.task_ready.store(
                (self.task_id + 1) % self.args.nreq,
                std::sync::atomic::Ordering::Relaxed,
            );
        };

        if self.req.is_some() {
            let mut all_done = true;

            for (j, req) in self.req.as_mut().unwrap().iter_mut().enumerate() {
                if req.is_some() {
                    let (done, _) = nccl_net::test(req.as_ref().unwrap()).unwrap();
                    if done {
                        trace!(
                            "{}  : task_id: {}, idx: {}, i: {}, j: {}, done, count: {}",
                            opname,
                            self.task_id,
                            idx,
                            i,
                            j,
                            self.count + 1
                        );
                        *req = None
                    } else {
                        all_done = false;
                    }
                }
            }

            if all_done {
                buf_ready.store(done_value, std::sync::atomic::Ordering::Relaxed);
                self.req = None;
                self.count += 1;
                self.next = (self.next + 1) % self.tasks.len();
                if self.count == self.try_count {
                    trace!("{} finished: task_id: {}", opname, self.task_id);
                    return true;
                }
            }
        }
        false
    }
}

fn comm_loop<T: Float>(
    args: &Args,
    comms: Vec<Comm>,
    tasks: Vec<Vec<(usize, Arc<AtomicUsize>, Vec<Arc<Mutex<AlignedBox<[T]>>>>)>>,
    is_recv: bool,
) {
    let task_readys = Arc::new(AtomicUsize::new(0)); // this is used to make task to be requrested in order
    let mut tasks = tasks
        .into_iter()
        .enumerate()
        .map(|(i, t)| Task::new(i, &comms, t, args, Arc::clone(&task_readys)))
        .collect::<Vec<_>>();

    let mut done = 0;
    let total_done = tasks.len();

    loop {
        for task in tasks.iter_mut() {
            if task.progress(is_recv) {
                done += 1;
                if done == total_done {
                    trace!("{} finished", if is_recv { "recv" } else { "send" });
                    return;
                }
            }
        }
    }
}

fn reduce_loop<T: Float>(
    task_id: usize,
    args: &Args,
    tasks: Vec<(
        usize,
        Arc<AtomicUsize>,
        Vec<Arc<Mutex<AlignedBox<[T]>>>>,
        Arc<AtomicUsize>,
        Vec<Arc<Mutex<AlignedBox<[T]>>>>,
    )>,
) {
    let mut count = 0;
    let mut work_mem = WorkingMemory::new(args.count, 1);
    let initial: T = T::from_usize(args.ring_rank).unwrap();
    let init = vec![initial; args.count];
    //    trace!("reduce_loop: len(tasks): {}", tasks.len());
    let nring = args.address.split(",").count();
    let try_count = args.try_count * tasks.len() / args.nreq / nring;
    loop {
        for (idx, (i, recv_ready, recv_bufs, send_ready, send_bufs)) in tasks.iter().enumerate() {
            while recv_ready.load(std::sync::atomic::Ordering::Relaxed) == 0
                || send_ready.load(std::sync::atomic::Ordering::Relaxed) == args.reduce_threads
            {
                hint::spin_loop();
            }

            trace!(
                "reduce: task_id: {}, idx: {}, i: {}, start",
                task_id,
                idx,
                i
            );
            send_bufs
                .iter()
                .zip(recv_bufs.iter())
                .for_each(|(send_buf, recv_buf)| {
                    let mut send = send_buf.lock().unwrap();
                    let recv = recv_buf.lock().unwrap();
                    let vecs = vec![init.as_ref(), recv.as_ref()];
                    send.reduce(&vecs, Some(&mut work_mem)).unwrap();
                });
            trace!(
                "reduce: task_id: {}, idx: {}, i: {}, done, count: {}",
                task_id,
                idx,
                i,
                count + 1
            );
            //                trace!(
            //                    "reduce: idx: {}, i: {}, done, count: {}, init: {:?}, recv: {:?}, send: {:?}",
            //                    idx,
            //                    i,
            //                    count + 1,
            //                    init,
            //                    recv.as_ref(),
            //                    send.as_ref()
            //                );
            recv_ready.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            send_ready.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            count += 1;
            if count == try_count {
                trace!("reduce finished");
                return;
            }
        }
    }
}

fn do_ring<T: Float + 'static>(args: Args, ch: usize, recvs: Vec<Comm>, sends: Vec<Comm>) {
    assert!(recvs.len() == sends.len());

    let size = args.count * std::mem::size_of::<T>();
    let args = Arc::new(args);

    let initial: T = T::from_usize(args.ring_rank).unwrap();

    let nreq = args.nreq;
    let nring = recvs.len();
    assert!(nring % args.reduce_threads == 0);

    let accs = (0..args.nrank)
        .map(|_| {
            (0..nreq)
                .map(|_| {
                    (0..nring)
                        .map(|_| {
                            let acc =
                                AlignedBox::<[T]>::slice_from_default(alignment(size), args.count)
                                    .unwrap();
                            Arc::new(Mutex::new(acc))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let bufs = (0..args.nrank)
        .map(|_| {
            (0..nreq)
                .map(|_| {
                    (0..nring)
                        .map(|_| {
                            let buf =
                                AlignedBox::<[T]>::slice_from_default(alignment(size), args.count)
                                    .unwrap();
                            Arc::new(Mutex::new(buf))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let init = (0..nring)
        .map(|_| {
            Arc::new(Mutex::new(
                AlignedBox::<[T]>::slice_from_value(alignment(size), args.count, initial).unwrap(),
            ))
        })
        .collect::<Vec<_>>();

    let reduce_send_atomics = (0..(args.nrank - 1))
        .map(|_| {
            (0..nreq)
                .map(|_| Arc::new(AtomicUsize::new(0)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let recv_send_atomics = (0..(args.nrank - 1))
        .map(|i| {
            let last = args.nrank - 2;
            let v = if i == last { args.reduce_threads } else { 0 };
            (0..nreq)
                .map(|_| Arc::new(AtomicUsize::new(v)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let recv_reduce_atomics = (0..(args.nrank - 1))
        .map(|_| {
            (0..nreq)
                .map(|_| Arc::new(AtomicUsize::new(0)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let recv_th = {
        let first = (0..args.nrank - 1).map(|i| {
            (0..nreq)
                .map(|j| {
                    let idx = (args.nrank + args.ring_rank + i) % args.nrank;
                    let bufs = bufs[idx][j]
                        .iter()
                        .map(|buf| Arc::clone(buf))
                        .collect::<Vec<_>>();
                    (idx, Arc::clone(&recv_reduce_atomics[i][j]), bufs)
                })
                .collect::<Vec<_>>()
        });
        let second = (0..args.nrank - 1).map(|i| {
            (0..nreq)
                .map(|j| {
                    let idx = (args.nrank - 1 + args.nrank + args.ring_rank + i) % args.nrank;
                    let accs = accs[idx][j]
                        .iter()
                        .map(|acc| Arc::clone(acc))
                        .collect::<Vec<_>>();
                    (idx, Arc::clone(&recv_send_atomics[i][j]), accs)
                })
                .collect::<Vec<_>>()
        });
        let mut tasks = first.chain(second).collect::<Vec<_>>();
        tasks = transpose(tasks);
        let args = Arc::clone(&args);
        std::thread::spawn(move || comm_loop(&args, recvs, tasks, true))
    };

    let reduce_ths = (0..args.reduce_threads)
        .map(|i| {
            let tasks = (0..args.nrank - 1)
                .map(|j| {
                    (0..nreq)
                        .map(|k| {
                            let idx = (args.nrank + args.ring_rank + i) % args.nrank;
                            let bufs = bufs[idx][k]
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| index % args.reduce_threads == i)
                                .map(|(_, buf)| Arc::clone(buf))
                                .collect::<Vec<_>>();
                            let accs = accs[idx][k]
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| index % args.reduce_threads == i)
                                .map(|(_, buf)| Arc::clone(buf))
                                .collect::<Vec<_>>();
                            (
                                idx,
                                Arc::clone(&recv_reduce_atomics[j][k]),
                                bufs,
                                Arc::clone(&reduce_send_atomics[j][k]),
                                accs,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect::<Vec<_>>();
            let args = Arc::clone(&args);
            std::thread::spawn(move || reduce_loop(i, &args, tasks))
        })
        .collect::<Vec<_>>();

    let send_th = {
        let first = (0..args.nrank - 1).map(|i| {
            (0..nreq)
                .map(|j| {
                    let idx = (args.nrank + args.ring_rank + i - 1) % args.nrank;
                    if i == 0 {
                        let init = init.iter().map(|i| Arc::clone(i)).collect::<Vec<_>>();
                        (idx, Arc::clone(&recv_send_atomics[args.nrank - 2][j]), init)
                    } else {
                        let accs = accs[idx][j]
                            .iter()
                            .map(|acc| Arc::clone(acc))
                            .collect::<Vec<_>>();
                        (idx, Arc::clone(&reduce_send_atomics[i - 1][j]), accs)
                    }
                })
                .collect::<Vec<_>>()
        });
        let second = (0..args.nrank - 1).map(|i| {
            (0..nreq)
                .map(|j| {
                    let idx = (args.nrank - 1 + args.nrank + args.ring_rank + i - 1) % args.nrank;
                    let accs = accs[idx][j]
                        .iter()
                        .map(|acc| Arc::clone(acc))
                        .collect::<Vec<_>>();
                    if i == 0 {
                        (
                            idx,
                            Arc::clone(&reduce_send_atomics[args.nrank - 2][j]),
                            accs,
                        )
                    } else {
                        (idx, Arc::clone(&recv_send_atomics[i - 1][j]), accs)
                    }
                })
                .collect::<Vec<_>>()
        });
        let mut tasks = first.chain(second).collect::<Vec<_>>();
        tasks = transpose(tasks);
        let args = Arc::clone(&args);
        std::thread::spawn(move || comm_loop(&args, sends, tasks, false))
    };

    let start = std::time::Instant::now();

    info!("ch: {} ring", ch);

    send_th.join().unwrap();
    info!("ch: {} send joined", ch);
    recv_th.join().unwrap();
    info!("ch: {} recv joined", ch);
    reduce_ths.into_iter().for_each(|h| h.join().unwrap());
    info!("ch: {} reduce joined", ch);

    // stop timer
    let elapsed = start.elapsed();
    print_stat(&args, &elapsed);

    //    // show contents of accs
    //    for (i, acc) in accs.iter().enumerate() {
    //        let a = acc[0].lock().unwrap();
    //        info!("accs[0][{}]: {:?}", i, a.as_ref());
    //        let a = acc[1].lock().unwrap();
    //        info!("accs[1][{}]: {:?}", i, a.as_ref());
    //    }
}

pub(crate) fn ring(args: Args) {
    let comms = args
        .address
        .split(',')
        .map(|addr| {
            let port = addr.split(':').last().unwrap().parse::<u16>().unwrap();

            let recvs = {
                let args = args.clone();
                std::thread::spawn(move || {
                    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).unwrap();
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
                let args = args.clone();
                let addr = addr.to_string();
                std::thread::spawn(move || {
                    let mut send = loop {
                        let res = TcpStream::connect(&addr);
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

            recvs.into_iter().zip(sends.into_iter()).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let comms = transpose(comms);

    let hs = comms
        .into_iter()
        .enumerate()
        .map(|(ch, comm)| {
            let args = args.clone();
            std::thread::spawn(move || {
                let (recvs, sends) = comm.into_iter().unzip();
                if args.data_type == DataType::F32 {
                    do_ring::<f32>(args, ch, recvs, sends);
                } else if args.data_type == DataType::F16 {
                    do_ring::<f16>(args, ch, recvs, sends);
                } else if args.data_type == DataType::BF16 {
                    do_ring::<bf16>(args, ch, recvs, sends);
                }
            })
        })
        .collect::<Vec<_>>();
    hs.into_iter().for_each(|h| h.join().unwrap());
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tests::initialize;
    use clap::Parser;

    #[test]
    fn test_ring() {
        initialize();
        (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let ring_rank = format!("{}", i + 1);
                    let address = format!("127.0.0.1:{},127.0.0.1:{}", 9090 + i, 9100 + i);
                    let args = Args::parse_from([
                        "--bench",
                        "--nrank",
                        "4",
                        "--reduce-threads",
                        "2",
                        "--address",
                        &address,
                        "--nreq",
                        "1", // when using socket plugin, concurrent recv/send requests doesn't work
                        "--ring-rank",
                        &ring_rank,
                    ]);
                    println!("{:?}", args);
                    ring(args);
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|h| h.join().unwrap());
    }
}
