FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu20.04 AS nccl

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -qy python3 openmpi-bin openmpi-common

COPY test test
COPY nccl_plugin nccl_plugin

RUN cd test && MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make && make install

FROM ghcr.io/rust-lang/rust:nightly-bullseye-slim AS optcast

COPY reduction_server reduction_server

COPY --from=nccl /usr/local/lib /usr/local/lib

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -qy clang libibverbs1

RUN cd reduction_server && cargo build -r

FROM optcast AS unittest

RUN cd reduction_server && cargo test --all

FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu20.04 AS final

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -qy --no-install-recommends openmpi-bin

COPY --from=nccl /usr/local/lib /usr/local/lib
COPY --from=nccl test/nccl-tests/build/*_perf /usr/local/bin/
COPY --from=optcast reduction_server/target/release/optcast-reduction-server /usr/local/bin/optcast-reduction-server

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV RUST_LOG=info
