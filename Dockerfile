# syntax=docker/dockerfile:1

########### Stage 1: builder ###########
FROM rust:1.75-slim AS builder

WORKDIR /app

# Copy the manifests first so the Docker layer cache speeds up dependency compilation
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# sqlx uses sqlite with rustls, so the build needs no extra system libraries
RUN cargo build --release -p yq-nova-server --bin yq-nova

########### Stage 2: runtime ###########
FROM debian:bookworm-slim

# The binary depends on glibc, which debian-slim already provides, so nothing extra is installed
COPY --from=builder /app/target/release/yq-nova /usr/local/bin/yq-nova

ENV YQ_NOVA_CONFIG=/etc/yq-nova/yq-nova.toml

RUN mkdir -p /etc/yq-nova /data

VOLUME ["/data"]

EXPOSE 7999

ENTRYPOINT ["yq-nova"]
CMD ["serve"]