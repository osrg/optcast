# Optcast

Optcast is an implementation of a reduction server written in Rust, specifically designed for enhanced performance in distributed machine learning environments utilizing the [NVIDIA NCCL library](https://github.com/NVIDIA/nccl) for the AllReduce collective operation. Although still in its prototype stage, Optcast has achieved a [50% speed improvement](./docs/eval_results.md) in NCCL's AllReduce operation under certain conditions.

## Ring-Allreduce vs Reduction Server

AllReduce is a crucial collective communication operation that is frequently used in distributed machine learning, and optimizing it is a vital concern.

In distributed machine learning environments with NVIDIA GPUs, NCCL commonly implements AllReduce using the Ring-AllReduce algorithm. Ring-AllReduce involves a ring-connected sequence of GPU nodes, each adding the gradients received from the preceding node to its own and passing them on to the next node. This algorithm is simple and efficiently utilizes network bandwidth in many environments, but it requires transferring twice the data volume of the gradients.

On the other hand, a reduction server is a server that simply receives the gradients from each GPU node, adds them together, and sends them back to all GPU nodes. Using reduction servers requires setting up separate servers for reduction, but it can theoretically double the processing speed of AllReduce since it only needs to transfer half the data volume compared to Ring-AllReduce.

For more detailed explanations of Ring-AllReduce and reduction servers, refer to [this blog article](https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance-reduction-server-vertex-ai?hl=en).

Furthermore, it's known that Ring Allreduce may encounter precision [issues](https://github.com/NVIDIA/nccl/issues/1026#issuecomment-1763933869), which, in principle, can be resolved by using reduction servers. Optcast has not yet evaluated the precision aspect, but it is an intriguing topic for future exploration.

## Features

- Implemented in Rust:
    - Optcast is developed in Rust. Given its extensive use of multi-threading, choosing Rust, which boasts [Fearless Concurrency](https://blog.rust-lang.org/2015/04/10/Fearless-Concurrency.html), was a logical decision.

- Support for Multiple Transport Protocols:
    - Communication between GPU servers and the reduction server uses [NCCL Net Plugin](https://github.com/NVIDIA/nccl/tree/master/ext-net), NCCL's communication primitive library. As a result, Optcast can operate in environments supporting sockets, InfiniBand, and RoCE, as enabled by NCCL.

- Utilization of Rust Portable SIMD:
    - FP32 addition operations are accelerated using SIMD instructions. Rust's [Portable SIMD](https://github.com/rust-lang/portable-simd?tab=readme-ov-file) is employed to ensure implementation is not dependent on specific CPUs.

- FP16 Support:
    - FP16 support is provided using [half-rs](https://github.com/starkat99/half-rs). While some CPUs can rapidly add FP16 natively, this is not yet common, so Optcast converts FP16 to FP32 for addition and then back to FP16. This conversion is efficiently handled by half-rs, utilizing fast, dedicated instructions when supported by the CPU.

## Similar Technologies

- [NVIDIA SHARP](https://docs.nvidia.com/networking/display/sharpv300)
    - NVIDIA SHARP optimizes AllReduce by performing reduction operations in the InfiniBand switch, eliminating the need for a separate reduction server. SHARP has its own NCCL plugin, so using it doesn't require changes to the application code. However, currently, SHARP is only available in InfiniBand environments.
- [Google Vertex AI Reduction Server](https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance-reduction-server-vertex-ai?hl=en)
    - This is a reduction server available in Google Vertex AI. According to the blog article, it approximately doubled the speed of each step in BERT's fine-tuning compared to NCCL Ring AllReduce.
- Parameter Servers (e.g., [BytePS](https://github.com/bytedance/byteps/tree/master)):
    - Parameter servers are similar to reduction servers but are not limited to optimizing AllReduce. They centrally manage the parameters of a learning model. Rather than being a collective communication acceleration component, they are offered as a framework for distributed machine learning and require modifications to the application code for use.

## Documentation

- [How to build Optcast](./docs/build.md)
- [How to evaluate Optcast](./docs/eval.md)
- [Evaluation Results](./docs/eval_results.md)

## License

Optcast is licensed under the [BSD 3-Clause License](./LICENSE).
