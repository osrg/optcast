## How to build Optcast

The Optcast Reduction Server communicates with NCCL applications through [the NCCL COLLNET plugin(libnccl-net.so)](https://github.com/NVIDIA/nccl/issues/320). To build the NCCL COLLNET plugin, follow these instructions:

Before you begin, you must install `NCCL` and `libibverb`.

```bash
$ cd $PATH_TO_REPO/nccl_plugin
$ ./autogen.sh
...
$ ./configure --with-cuda=/usr/local/cuda
$ make
...
$ sudo make install
$ ls /usr/local/lib/libnccl-net.* 
/usr/local/lib/libnccl-net.a  /usr/local/lib/libnccl-net.la  /usr/local/lib/libnccl-net.so  /usr/local/lib/libnccl-net.so.0  /usr/local/lib/libnccl-net.so.0.0.0
$
```

Next, we will build the Optcast Reduction Server. Since the Optcast Reduction Server is implemented in Rust, it can be easily built using Cargo.
Please note that building Optcast requires nightly Rust, as it utilizes the `c_variadic`, `portable_simd`, and `min_specialization` features, which are currently unstable.

```bash
$ cd $PATH_TO_REPO/reduction_server
$ cargo build -r
$ ./target/release/optcast-reduction-server -h
Usage: optcast-reduction-server [OPTIONS]

Options:
  -v, --verbose                          
  -c, --client                           
  -p, --port <PORT>                      [default: 8918]
  -a, --address <ADDRESS>                [default: 0.0.0.0]
      --count <COUNT>                    [default: 1024]
      --try-count <TRY_COUNT>            [default: 100]
      --reduce-threads <REDUCE_THREADS>  threads per reduce job [default: 2]
      --reduce-jobs <REDUCE_JOBS>        [default: 2]
      --recv-threads <RECV_THREADS>      [default: 0]
      --send-threads <SEND_THREADS>      [default: 0]
      --nrank <NRANK>                    [default: 1]
      --data-type <DATA_TYPE>            [default: f32] [possible values: f32, f16]
  -h, --help                             Print help
$
```
