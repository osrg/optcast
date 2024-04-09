# AWS Elastic Fabric Adapter (EFA) with Optcast

This document explains the steps to run Optcast with [AWS Elastic Fabric Adapter (EFA)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html).

AWS EFA is an OS bypass technology that accelerates communication for HPC applications, enabling fast RDMA between EC2 instances that support EFA.

By running the Optcast Reduction Server on less expensive CPU instances rather than GPU instances, and accelerating collective communication between GPU instances running distributed machine learning applications, it is possible not only to improve the execution speed of distributed machine learning applications but also to reduce the total cost.

For example, the on-demand price of a CPU instance that supports EFA, `c7gn.16xlarge`, is approximately [$4](https://instances.vantage.sh/aws/ec2/c7gn.16xlarge), whereas the on-demand price for a GPU instance that supports EFA, specifically `p4d.24xlarge`, is about [$32](https://instances.vantage.sh/aws/ec2/p4d.24xlarge).

`c7gn.16xlarge` has a bandwidth of 200Gbps, and `p4d.24xlarge` has a bandwidth of 400Gbps. Therefore, to match one `p4d.24xlarge` instance, it would be necessary to set up 2 `c7gn.16xlarge` instances.

Let the reduction rate in the application's execution time by using Optcast be denoted as $x$, and when the costs are equivalent with or without the use of Optcast, $x$ can be determined as follows.

$32 = (32 + 4*2) * (1-x)$, $x = 0.2$

In essence, if using Optcast can reduce the application's execution time by more than 20%, it's not only the execution time that can be minimized, but also the EC2 usage costs. Considering that collective communication becomes a bottleneck in large-scale distributed machine learning, such a scenario is realistic.

Unfortunately, currently, EFA traffic between P4d/P4de/DL1 instances and other instance types is [not supported](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-limits), so the configuration described above cannot be implemented at this time. Here, only `c7gn.16xlarge` instances are used, and the evaluation of the Reduction Server and Ring AllReduce in the EFA environment is performed using the client mode and Ring AllReduce mode, which are test functionalities of the Optcast Reduction Server.

## How to run Optcast with EFA (comparison with Ring AllReduce)

Since the Optcast NCCL plugin does not yet support EFA, to use EFA with Optcast, it is necessary to use the official EFA NCCL plugin, [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl).

Download https://github.com/aws/aws-ofi-nccl and build it according to the README.md.

Here, we assume that AWS OFI NCCL is installed in `/usr/local/lib` on all nodes running Optcast.

You can then evaluate using `test/run.py`. Here, we are setting up 8 instances of `c7gn.16xlarge` for evaluation. Prepare a `config.yaml` as follows,

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

Specify the `/usr/local/lib` where AWS OFI NCCL is installed with `--nccl-plugin-path`, and run with the `--no-gpu` option since there are no GPU instances in this environment. With the `--no-gpu` option, `run.py` uses Optcast's client feature for the evaluation instead of `nccl-tests`. To use the Optcast Reduction Server, specify `--type optcast`.

```bash
$ python run.py --nccl-plugin-path /usr/local/lib --no-gpu  --chunksize 8M --type optcast
[optcast7] [2024-04-09T08:37:07.459385392Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 2097152, try_count: 1000 #
[optcast7] [2024-04-09T08:37:07.459400409Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 126.77Gbps #
[optcast5] [2024-04-09T08:37:07.459568713Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 2097152, try_count: 1000 #
[optcast5] [2024-04-09T08:37:07.459571605Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 124.01Gbps #
[optcast8] [2024-04-09T08:37:07.459587838Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 2097152, try_count: 1000 #
[optcast8] [2024-04-09T08:37:07.459602362Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 129.74Gbps #
[optcast6] [2024-04-09T08:37:07.459685847Z INFO  optcast_reduction_server::utils] type: agg, nchannel: 1, nsplit: 4, nreq: 4, count: 2097152, try_count: 1000 #
[optcast6] [2024-04-09T08:37:07.459699786Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 133.05Gbps #
client stats:
  send len: 2000, avg: 2.81, sd: 0.79, median: 2.78, min: 1.25, max: 5.49
  recv len: 2000, avg: 6.37, sd: 0.93, median: 6.36, min: 3.60, max: 8.79

server stats:
  recv len: 2000, avg: 1.65, sd: 0.27, median: 1.60, min: 0.85, max: 2.89
  reduce len: 1000, avg: 0.84, sd: 0.03, median: 0.83, min: 0.78, max: 0.94
  send len: 2000, avg: 1.65, sd: 0.27, median: 1.61, min: 0.85, max: 2.76
```

If you check `log/client.log`, you can see EFA is used for communication.

```bash
head -n 10 log/client.log 
[optcast5] [2024-04-09T08:18:39.985173130Z INFO  optcast_reduction_server::nccl_net] [nccl_net_ofi_init:49] NET/OFI Initializing aws-ofi-nccl GitHub-dev
[optcast5] [2024-04-09T08:18:39.985192602Z INFO  optcast_reduction_server::nccl_net] [nccl_net_ofi_create_plugin:746] NET/OFI Initializing aws-ofi-nccl GitHub-dev
[optcast5] [2024-04-09T08:18:39.985197418Z INFO  optcast_reduction_server::nccl_net] [nccl_net_ofi_create_plugin:750] NET/OFI Using Libfabric version 1.20
[optcast5] [2024-04-09T08:18:39.985258120Z INFO  optcast_reduction_server::nccl_net] [nccl_net_ofi_create_plugin:776] NET/OFI Using CUDA driver version 12030
[optcast5] [2024-04-09T08:18:39.985262508Z INFO  optcast_reduction_server::nccl_net] [platform_init:343] NET/OFI Configuring AWS-specific options
[optcast5] [2024-04-09T08:18:39.985285577Z INFO  optcast_reduction_server::nccl_net] [platform_init:354] NET/OFI Setting provider_filter to efa
[optcast5] [2024-04-09T08:18:39.985288356Z INFO  optcast_reduction_server::nccl_net] [platform_init:392] NET/OFI Setting FI_EFA_FORK_SAFE
[optcast5] [2024-04-09T08:18:39.985293420Z INFO  optcast_reduction_server::nccl_net] [platform_init:439] NET/OFI Setting NCCL_NVLSTREE_MAX_CHUNKSIZE to 512KiB
[optcast5] [2024-04-09T08:18:39.985296733Z INFO  optcast_reduction_server::nccl_net] [platform_init:492] NET/OFI Internode latency set at <unknown: .>
```

Next, let's evaluate Ring AllReduce for comparison. By specifing `--type ring`, `run.py` uses Optcast's Ring AllReduce implementation for the evaluation.

```bash
$ python run.py --nccl-plugin-path /usr/local/lib --no-gpu  --chunksize 8M --type ring
[optcast8] [2024-04-09T08:35:03.119882121Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 2097152, try_count: 1000 #
[optcast8] [2024-04-09T08:35:03.119884667Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 97.20Gbps #
[optcast7] [2024-04-09T08:35:03.120915762Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 2097152, try_count: 1000 #
[optcast7] [2024-04-09T08:35:03.120919400Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 96.15Gbps #
[optcast6] [2024-04-09T08:35:03.122185356Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 2097152, try_count: 1000 #
[optcast6] [2024-04-09T08:35:03.122189332Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 96.67Gbps #
[optcast5] [2024-04-09T08:35:03.122789003Z INFO  optcast_reduction_server::utils] type: ring, nchannel: 1, nsplit: 2, nreq: 4, nrank: 4, reduce_ths: 2, count: 2097152, try_count: 1000 #
[optcast5] [2024-04-09T08:35:03.122791856Z INFO  optcast_reduction_server::utils] size: 32.00MB, bandwidth: 96.94Gbps #
client stats:
  send len: 3000, avg: 1.72, sd: 0.64, median: 1.59, min: 0.77, max: 4.55
  recv len: 3000, avg: 2.97, sd: 0.71, median: 2.96, min: 0.77, max: 4.85
  reduce len: 1500, avg: 0.83, sd: 0.01, median: 0.83, min: 0.78, max: 0.86
```

As you can see, the performance of AllReduce using Optcast Reduction Server is better than Ring AllReduce (~= 30%).

Since `c7gn.16xlarge` has a network bandwidth of 200Gbps, the maximum performance when using a Reduction Server is 200Gbps. [In the case of Ring AllReduce](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce), with 4 nodes, it becomes $200 * 4 / 2(4-1)$, which is 133.33Gbps, and from the above results, it can be seen that the network bandwidth is not yet fully utilized. Optimizing to fully utilize the network bandwidth under the EFA environment is a challenge for the future.

For detailed instructions on how to use `run.py`, please refer to [eval.md](./eval.md).
