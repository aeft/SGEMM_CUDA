#!/usr/bin/env bash

set -euo pipefail

# This scripts runs the ./sgemm binary for all exiting kernels, and logs
# the outputs to text files in benchmark_results/. Then it calls
# the plotting script

mkdir -p benchmark_results

for kernel in {0..10}; do
    echo ""
    ./build/sgemm $kernel | tee "benchmark_results/${kernel}_output.txt"
    sleep 2
done

# Use venv python if available, otherwise fallback to system python3
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python plot_benchmark_results.py
else
    python3 plot_benchmark_results.py
fi
