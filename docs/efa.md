# AWS Elastic Fabric Adapter (EFA) with Optcast

This document explains the steps to run Optcast with [AWS Elastic Fabric Adapter (EFA)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html).

AWS EFA is an OS bypass technology that accelerates communication for HPC applications, enabling fast RDMA between EC2 instances that support EFA.

By running the Optcast Reduction Server on less expensive CPU instances rather than GPU instances, and accelerating collective communication between GPU instances running distributed machine learning applications, it is possible not only to improve the execution speed of distributed machine learning applications but also to reduce the total cost.

For example, the on-demand price of a CPU instance that supports EFA, `c5n.9xlarge`, is approximately [$2](https://instances.vantage.sh/aws/ec2/c5n.9xlarge), whereas the on-demand price for a GPU instance that supports EFA, specifically `p4d.24xlarge`, is about [$32](https://instances.vantage.sh/aws/ec2/p4d.24xlarge).

`c5n.9xlarge` has a bandwidth of 50Gbps, and `p4d.24xlarge` has a bandwidth of 400Gbps. Therefore, to match one `p4d.24xlarge` instance, it would be necessary to set up 8 `c5n.9xlarge` instances.

Assuming that using Optcast can halve the total application execution time, the total cost could be reduced by 25%: $32 vs $24 = (32 + 2*8)/2.

Currently, EFA traffic between P4d/P4de/DL1 instances and other instance types is [not supported](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-limits), so the configuration described above cannot be implemented at this time. Here, only `c5n.9xlarge` instances are used, and the evaluation of the Reduction Server and Ring AllReduce in the EFA environment is performed using the client mode and Ring AllReduce mode, which are test functionalities of the Optcast Reduction Server.

## How to run Optcast with EFA

Since the Optcast NCCL plugin does not yet support EFA, to use EFA with Optcast, it is necessary to use the official EFA NCCL plugin, [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl).

Download https://github.com/aws/aws-ofi-nccl and build it according to the README.md.

Here, we assume that AWS OFI NCCL is installed in `/usr/local/lib` on all nodes running Optcast.

You can then evaluate using `test/run.py`. Here, we are setting up 8 instances of `c5n.9xlarge` for evaluation. Prepare a `config.yaml` as follows,

```bash
$ cat config.yaml
servers:
  - name: optcast1
    port: 8080
  - name: optcast2
    port: 8080
  - name: optcast3
    port: 8080
  - name: optcast4
    port: 8080
clients:
  - name: optcast5
  - name: optcast6
  - name: optcast7
  - name: optcast8
``` 

Specify the `/usr/local/lib` where AWS OFI NCCL is installed with `--nccl-plugin-path`, and run with the `--no-gpu` option since there are no GPU instances in this environment. To use the Optcast Reduction Server, specify `--type optcast`.

```bash
$ python3 run.py --nccl-plugin-path /usr/local/lib --no-gpu --type optcast
client stderr: [ip-172-31-32-6:84433] Warning: could not find environment variable "LD_LIBRARY_PATH"
server stats:
  recv len: 2000, avg: 0.48, sd: 0.11, median: 0.49, min: 0.22, max: 1.63
  reduce len: 1000, avg: 0.09, sd: 0.01, median: 0.09, min: 0.07, max: 0.13
  send len: 2000, avg: 0.57, sd: 0.12, median: 0.57, min: 0.23, max: 1.25

[ip-172-31-32-16] [2024-03-11T09:35:20.613735237Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 131072, try_count: 1000 #
[ip-172-31-32-16] [2024-03-11T09:35:20.613739443Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 22.98Gbps #
[ip-172-31-32-156] [2024-03-11T09:35:20.613352791Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 131072, try_count: 1000 #
[ip-172-31-32-156] [2024-03-11T09:35:20.613386752Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 20.97Gbps #
[ip-172-31-32-154] [2024-03-11T09:35:20.613555708Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 131072, try_count: 1000 #
[ip-172-31-32-154] [2024-03-11T09:35:20.613572396Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 17.90Gbps #
[ip-172-31-32-95] [2024-03-11T09:35:20.613632963Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 131072, try_count: 1000 #
[ip-172-31-32-95] [2024-03-11T09:35:20.613651423Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 19.31Gbps #
client stats:
  send len: 2000, avg: 0.55, sd: 0.26, median: 0.46, min: 0.24, max: 2.18
  recv len: 2000, avg: 1.67, sd: 0.67, median: 1.42, min: 0.67, max: 3.37
```

To evaluate Ring AllReduce, specify `--type ring`.

```bash
$ python3 run.py --nccl-plugin-path /usr/local/lib --no-gpu --type ring
client stderr: [ip-172-31-32-6:84894] Warning: could not find environment variable "LD_LIBRARY_PATH"
[ip-172-31-32-16] [2024-03-11T09:39:14.988766319Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 131072, try_count: 1000 #
[ip-172-31-32-16] [2024-03-11T09:39:14.988769309Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 12.25Gbps #
[ip-172-31-32-154] [2024-03-11T09:39:14.990050901Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 131072, try_count: 1000 #
[ip-172-31-32-154] [2024-03-11T09:39:14.990053586Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 12.34Gbps #
[ip-172-31-32-156] [2024-03-11T09:39:14.990524388Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 131072, try_count: 1000 #
[ip-172-31-32-156] [2024-03-11T09:39:14.990527071Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 12.13Gbps #
[ip-172-31-32-95] [2024-03-11T09:39:14.993997739Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 131072, try_count: 1000 #
[ip-172-31-32-95] [2024-03-11T09:39:14.994000596Z INFO  optcast_reduction_server::utils] size: 2.00MB, bandwidth: 12.19Gbps #
client stats:
  send len: 3000, avg: 0.95, sd: 0.49, median: 0.81, min: 0.25, max: 3.06
  recv len: 3000, avg: 1.19, sd: 0.66, median: 1.02, min: 0.21, max: 3.86
  reduce len: 1500, avg: 0.09, sd: 0.01, median: 0.09, min: 0.07, max: 0.14
```

For detailed instructions on how to use `run.py`, please refer to [eval.md](./eval.md).
