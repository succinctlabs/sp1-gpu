#!/bin/bash

N=6

for ((i=1; i<=N; i++)); do
    echo "Run $i of $N"
    if ! bash run_s3.sh reth compress; then
        echo "Run $i failed"
        echo "Summary: 1 failure out of $i runs"
        echo "Success rate: $(( (i-1)*100/i ))%"
        exit 1
    fi
done

echo "Summary: 0 failures out of $N runs"
echo "Success rate: 100%"
