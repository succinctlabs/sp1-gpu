#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <s3_path> <core|compress|shrink|wrap>"
    exit 1
fi

s3_path=$1
stage=$2

# Download files from S3
aws s3 cp s3://sp1-testing-suite/$s3_path/program.bin /tmp/program.bin
aws s3 cp s3://sp1-testing-suite/$s3_path/stdin.bin /tmp/stdin.bin

# Set environment variables
export RUST_LOG=debug
export FIX_CORE_SHAPES=false
export FIX_RECURSION_SHAPES=true

# Run moongate-perf
RUST_BACKTRACE=full cargo run -p moongate-perf --release -- --program-path /tmp/program.bin --stdin-path /tmp/stdin.bin --stage $stage
