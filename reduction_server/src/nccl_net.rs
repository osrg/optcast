/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::io::{BufWriter, Write};

use aligned_box::AlignedBox;
use log::{log, log_enabled};

use nccl_net_sys as ffi;

unsafe extern "C" fn logfn(
    level: ffi::ncclDebugLogLevel::Type,
    _: ::std::os::raw::c_ulong,
    cfile: *const ::std::os::raw::c_char,
    line: ::std::os::raw::c_int,
    cfmt: *const ::std::os::raw::c_char,
    mut ap: ...
) {
    let fmt = std::ffi::CStr::from_ptr(cfmt).to_str().unwrap();

    let (enabled, level) = match level {
        ffi::ncclDebugLogLevel::NCCL_LOG_WARN => (log_enabled!(log::Level::Warn), log::Level::Warn),
        ffi::ncclDebugLogLevel::NCCL_LOG_INFO => (log_enabled!(log::Level::Info), log::Level::Info),
        ffi::ncclDebugLogLevel::NCCL_LOG_ABORT => {
            (log_enabled!(log::Level::Error), log::Level::Error)
        }
        ffi::ncclDebugLogLevel::NCCL_LOG_TRACE => {
            (log_enabled!(log::Level::Trace), log::Level::Trace)
        }
        _ => (false, log::Level::Info),
    };
    if !enabled {
        return;
    }

    let out = Vec::new();
    let mut out = BufWriter::new(out);

    let file = std::ffi::CStr::from_ptr(cfile).to_str().unwrap();
    write!(out, "[{}:{}] ", file, line).unwrap();

    if fmt.find('%').is_none() {
        write!(out, "{}", fmt).unwrap();
    } else {
        let mut i = 0;
        while let Some(j) = fmt[i..].find('%') {
            write!(out, "{}", &fmt[i..i + j]).unwrap();
            i += j;
            let mut long = 0;
            loop {
                match fmt[i + 1..].chars().next().unwrap() {
                    'l' => {
                        long += 1;
                    }
                    'd' => {
                        let v = match long {
                            0 => ap.arg::<::std::os::raw::c_int>() as i64,
                            1 => ap.arg::<::std::os::raw::c_long>() as i64,
                            _ => ap.arg::<::std::os::raw::c_longlong>() as i64,
                        };
                        write!(out, "{}", v).unwrap();
                        break;
                    }
                    'x' => {
                        let v = match long {
                            0 => ap.arg::<::std::os::raw::c_int>() as i64,
                            1 => ap.arg::<::std::os::raw::c_long>() as i64,
                            _ => ap.arg::<::std::os::raw::c_longlong>() as i64,
                        };
                        write!(out, "{:x}", v).unwrap();
                        break;
                    }
                    's' => {
                        let v = ap.arg::<*const ::std::os::raw::c_char>();
                        let v = std::ffi::CStr::from_ptr(v).to_str().unwrap();
                        write!(out, "{}", v).unwrap();
                        break;
                    }
                    'u' => {
                        let v = match long {
                            0 => ap.arg::<::std::os::raw::c_uint>() as i64,
                            1 => ap.arg::<::std::os::raw::c_ulong>() as i64,
                            _ => ap.arg::<::std::os::raw::c_ulonglong>() as i64,
                        };
                        write!(out, "{}", v).unwrap();
                        break;
                    }
                    'p' => {
                        let v = ap.arg::<*const ::std::os::raw::c_void>();
                        write!(out, "{:?}", v).unwrap();
                        break;
                    }
                    'f' => {
                        let v = ap.arg::<::std::os::raw::c_double>();
                        write!(out, "{}", v).unwrap();
                        break;
                    }
                    '%' => {
                        write!(out, "%").unwrap();
                        break;
                    }
                    _ => {
                        // raise error
                        write!(out, "<unknown: {}>", fmt[i + 1..].chars().next().unwrap()).unwrap();
                        break;
                    }
                }
                i += 1;
            }
            i += 2;
        }
    }
    log!(
        level,
        "{}",
        String::from_utf8(out.into_inner().unwrap()).unwrap()
    );
}

#[derive(Debug)]
enum CommType {
    Listen,
    Send,
    Recv,
}

#[derive(Debug)]
pub(crate) struct Comm {
    ptr: *mut std::ffi::c_void,
    r#type: CommType,
}

impl Drop for Comm {
    fn drop(&mut self) {
        println!("drop comm: {:?}", self.r#type);
        let ret = match self.r#type {
            CommType::Listen => unsafe { ffi::ncclNetPlugin_v6.closeListen.unwrap()(self.ptr) },
            CommType::Send => unsafe { ffi::ncclNetPlugin_v6.closeSend.unwrap()(self.ptr) },
            CommType::Recv => unsafe { ffi::ncclNetPlugin_v6.closeRecv.unwrap()(self.ptr) },
        };
        if ret != ffi::ncclResult_t::ncclSuccess {
            panic!("failed to close comm: {:?}", ret);
        }
    }
}

unsafe impl Send for Comm {}

#[derive(Debug)]
pub(crate) struct MemoryHandle<'a> {
    ptr: *mut std::ffi::c_void,
    comm: &'a Comm,
}

impl Drop for MemoryHandle<'_> {
    fn drop(&mut self) {
        println!("drop mh");
        let ret = unsafe { ffi::ncclNetPlugin_v6.deregMr.unwrap()(self.comm.ptr, self.ptr) };
        if ret != ffi::ncclResult_t::ncclSuccess {
            panic!("failed to dereg_mr: {:?}", ret);
        }
    }
}

unsafe impl Send for MemoryHandle<'_> {}

#[derive(Debug)]
pub(crate) struct Request {
    ptr: *mut std::ffi::c_void,
}

unsafe impl Send for Request {}

pub(crate) fn init() {
    unsafe {
        ffi::ncclNetPlugin_v6.init.unwrap()(Some(logfn));
    }
}

pub(crate) fn listen() -> Result<(Comm, Vec<u8>), ffi::ncclResult_t::Type> {
    let mut handle = [0u8; ffi::NCCL_NET_HANDLE_MAXSIZE as usize];
    let mut lcomm = std::ptr::null_mut();
    let lcomm_ptr = &mut lcomm;
    let ret = unsafe {
        ffi::ncclNetPlugin_v6.listen.unwrap()(
            0,
            handle.as_mut_ptr() as *mut std::ffi::c_void,
            lcomm_ptr,
        )
    };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    let handle = handle.to_vec();
    Ok((
        Comm {
            ptr: lcomm,
            r#type: CommType::Listen,
        },
        handle,
    ))
}

pub(crate) fn connect(handle: &[u8]) -> Result<Option<Comm>, ffi::ncclResult_t::Type> {
    let mut scomm = std::ptr::null_mut();
    let scomm_ptr = &mut scomm;
    let handle_ptr: *mut std::ffi::c_void = handle.as_ptr() as *mut std::ffi::c_void;
    let ret = unsafe { ffi::ncclNetPlugin_v6.connect.unwrap()(0, handle_ptr, scomm_ptr) };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    Ok(if scomm.is_null() {
        None
    } else {
        Some(Comm {
            ptr: scomm,
            r#type: CommType::Send,
        })
    })
}

pub(crate) fn accept(comm: &Comm) -> Result<Option<Comm>, ffi::ncclResult_t::Type> {
    let mut rcomm = std::ptr::null_mut();
    let rcomm_ptr = &mut rcomm;
    let ret = unsafe { ffi::ncclNetPlugin_v6.accept.unwrap()(comm.ptr, rcomm_ptr) };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    Ok(if rcomm.is_null() {
        None
    } else {
        Some(Comm {
            ptr: rcomm,
            r#type: CommType::Recv,
        })
    })
}

pub(crate) fn reg_mr<'a, T>(
    comm: &'a Comm,
    data: &AlignedBox<[T]>,
) -> Result<MemoryHandle<'a>, ffi::ncclResult_t::Type> {
    let mut mh = std::ptr::null_mut();
    let mh_ptr = &mut mh;
    let ptr = data.as_ptr() as *mut ::std::os::raw::c_void;
    let len = (data.len() * std::mem::size_of::<T>()) as ::std::os::raw::c_int;

    println!("ptr: {:p}, len: {}", ptr, len);
    let ret = unsafe {
        ffi::ncclNetPlugin_v6.regMr.unwrap()(
            comm.ptr,
            ptr,
            len,
            ffi::NCCL_PTR_HOST as ::std::os::raw::c_int,
            mh_ptr,
        )
    };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    Ok(MemoryHandle {
        ptr: mh,
        comm: comm,
    })
}

pub(crate) fn isend<T>(
    comm: &Comm,
    mhandle: &MemoryHandle,
    data: &AlignedBox<[T]>,
    tag: i32,
) -> Result<Option<Request>, ffi::ncclResult_t::Type> {
    let mut request = std::ptr::null_mut();
    let request_ptr = &mut request;
    let ret = unsafe {
        ffi::ncclNetPlugin_v6.isend.unwrap()(
            comm.ptr,
            data.as_ptr() as *mut ::std::os::raw::c_void,
            (data.len() * std::mem::size_of::<T>()) as ::std::os::raw::c_int,
            tag,
            mhandle.ptr,
            request_ptr,
        )
    };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    Ok(if request.is_null() {
        None
    } else {
        Some(Request { ptr: request })
    })
}

pub(crate) fn irecv<T>(
    comm: &Comm,
    mhandle: &MemoryHandle,
    data: &mut AlignedBox<[T]>,
    tag: i32,
) -> Result<Option<Request>, ffi::ncclResult_t::Type> {
    let mut request = std::ptr::null_mut();
    let request_ptr = &mut request;
    let data_ptr = &mut data.as_mut_ptr() as *mut _;
    let len_ptr =
        &mut ((data.len() * std::mem::size_of::<T>()) as i32) as *mut ::std::os::raw::c_int;
    let mhandle_ptr = &mhandle.ptr as *const _ as *mut _;
    let ret = unsafe {
        ffi::ncclNetPlugin_v6.irecv.unwrap()(
            comm.ptr,
            1,
            data_ptr as *mut *mut ::std::os::raw::c_void,
            len_ptr,
            &tag as *const _ as *mut ::std::os::raw::c_int,
            mhandle_ptr,
            request_ptr,
        )
    };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    Ok(if request.is_null() {
        None
    } else {
        Some(Request { ptr: request })
    })
}

pub(crate) fn test(request: &Request) -> Result<(bool, usize), ffi::ncclResult_t::Type> {
    let mut done = 0;
    let done_ptr = &mut done;
    let mut sizes = 0;
    let sizes_ptr = &mut sizes;
    let ret = unsafe { ffi::ncclNetPlugin_v6.test.unwrap()(request.ptr, done_ptr, sizes_ptr) };
    if ret != ffi::ncclResult_t::ncclSuccess {
        return Err(ret);
    }
    Ok(((done != 0), sizes as usize))
}
