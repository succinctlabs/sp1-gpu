#!/bin/bash

N=25
failures=0

for ((i=1; i<=N; i++)); do
    echo "Run $i of $N"
    if ! bash perf/run_s3.sh reth compress; then
        ((failures++))
        echo "Run $i failed"
        break
    fi
done

echo "Summary: $failures, N: $N"
