# Build stage
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --bin batchline

# Runtime stage: just the binary and the model
FROM debian:bookworm-slim
COPY --from=build /src/target/release/batchline /usr/local/bin/batchline
COPY models /models
USER nobody
EXPOSE 8080
ENTRYPOINT ["batchline"]
CMD ["--model", "/models/resnet14.onnx", "--addr", "0.0.0.0:8080"]
