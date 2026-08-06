# syntax=docker/dockerfile:1

########### Stage 1: builder ###########
FROM rust:1.75-slim AS builder

WORKDIR /app

# 先拷贝清单文件，利用 Docker 层缓存加速依赖编译
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# sqlx 使用 sqlite（rustls），构建无需额外系统库
RUN cargo build --release -p yq-nova-server --bin yq-nova

########### Stage 2: runtime ###########
FROM debian:bookworm-slim

# 二进制依赖 glibc，debian-slim 自带，无需额外安装
COPY --from=builder /app/target/release/yq-nova /usr/local/bin/yq-nova

ENV YQ_NOVA_CONFIG=/etc/yq-nova/yq-nova.toml

RUN mkdir -p /etc/yq-nova /data

VOLUME ["/data"]

EXPOSE 7999

ENTRYPOINT ["yq-nova"]
CMD ["serve"]