# moongate

## Profiling

### Jaeger

Setup Jaeger:
```
sudo docker run -it --rm -d -p4318:4318 -p4317:4317 -p16686:16686 jaegertracing/all-in-one:latest
```

Run a benchmark:
```
RUST_LOG=debug cargo run --release -p moongate-perf -- --program fibonacci
```

### Nvidia Nsight Systems

Run a benchmark:
```
RUST_LOG="debug" nsys profile --trace=cuda,nvtx cargo run --release -p moongate-perf -- --program fibonacci --trace nvtx 
```

## Server

Build the server image:
```
sudo docker build -f Dockerfile.server -t moongate-server .
```

Run the server:
```
sudo docker run -e "RUST_LOG=debug" -p 3000:3000 --rm --runtime=nvidia --gpus all moongate-server
```