#!/bin/bash

set -e

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <s3_path> <core|compress|shrink|wrap>"
    exit 1
fi

s3_path=$1
stage=$2

# Download files from S3
aws s3 cp s3://sp1-testing-suite/$s3_path/program.bin program.bin
aws s3 cp s3://sp1-testing-suite/$s3_path/stdin.bin stdin.bin

# Set environment variables
export RUST_LOG=debug
export FIX_CORE_SHAPES=true
export FIX_RECURSION_SHAPES=true
export SHARD_BATCH_SIZE=1
export SP1_ALLOW_DEPRECATED_HOOKS=true

# Run moongate-perf
RUST_BACKTRACE=full cargo run -p moongate-perf --release -- --program-path program.bin --stdin-path stdin.bin --stage $stage --trace nvtx

# Remove the downloaded files
rm program.bin stdin.bin 