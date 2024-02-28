FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 AS nccl

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -qy python3 openmpi-bin openmpi-common libibverbs-dev libopenmpi-dev autoconf libtool

COPY test test

RUN cd test && make build-nccl
RUN cd test && MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make build-nccl-tests

COPY nccl_plugin nccl_plugin

RUN cd test && make build-nccl-plugin
RUN cd test && make install

FROM ubuntu:22.04 AS optcast

RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Rust install
ENV RUST_HOME /usr/local/lib/rust
ENV RUSTUP_HOME ${RUST_HOME}/rustup
ENV CARGO_HOME ${RUST_HOME}/cargo
RUN mkdir /usr/local/lib/rust && \
    chmod 0755 $RUST_HOME
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > ${RUST_HOME}/rustup.sh \
    && chmod +x ${RUST_HOME}/rustup.sh \
    && ${RUST_HOME}/rustup.sh -y --default-toolchain nightly --no-modify-path
ENV PATH $PATH:$CARGO_HOME/bin

COPY --from=nccl /usr/local/lib /usr/local/lib

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -qy clang libibverbs1

COPY reduction_server reduction_server

RUN cd reduction_server && cargo build -r

FROM optcast AS unittest

ENV RUST_LOG=info
RUN cd reduction_server && cargo test --all -- --nocapture

FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 AS final

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -qy --no-install-recommends openmpi-bin

COPY --from=nccl /usr/local/lib /usr/local/lib
COPY --from=nccl test/nccl-tests/build/*_perf /usr/local/bin/
COPY --from=optcast reduction_server/target/release/optcast-reduction-server /usr/local/bin/optcast-reduction-server

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV RUST_LOG=info
