#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32)
TM_VALUES=(4 8 16)
TN_VALUES=(4 8 16)
BM_VALUES=(64 128)
BN_VALUES=(64 128)
NUM_THREADS=256

cd "$(dirname "$0")"
cd ..

RUNNER="src_practice/runner.cu"
OUTPUT="benchmark_results/kernel_9_practice_autotune_results.txt"

# Create output directory if needed
mkdir -p "$(dirname $OUTPUT)"

# Clear the output file
echo "" > $OUTPUT

TOTAL_CONFIGS="$(( ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          echo ""
          CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

          config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn"

          # skip configurations that don't fulfill preconditions
          # NUM_THREADS * 4 must be divisible by BK (vectorized A load)
          if [[ $(( ($NUM_THREADS * 4) % $bk )) -ne 0 ]]; then
            echo "Skipping $config: (NUM_THREADS * 4) % BK != 0"
            continue
          fi
          # NUM_THREADS * 4 must be divisible by BN (vectorized B load)
          if [[ $(( ($NUM_THREADS * 4) % $bn )) -ne 0 ]]; then
            echo "Skipping $config: (NUM_THREADS * 4) % BN != 0"
            continue
          fi
          # BN must be divisible by (16 * TN) for WN tiling
          if [[ $(( $bn % (16 * $tn) )) -ne 0 ]]; then
            echo "Skipping $config: BN % (16 * TN) != 0"
            continue
          fi
          # BM must be divisible by (16 * TM) for WM tiling
          if [[ $(( $bm % (16 * $tm) )) -ne 0 ]]; then
            echo "Skipping $config: BM % (16 * TM) != 0"
            continue
          fi
          # (BM * BK) must be divisible by (4 * NUM_THREADS) for vectorized A load loop
          if [[ $(( ($bm * $bk) % (4 * $NUM_THREADS) )) -ne 0 ]]; then
            echo "Skipping $config: (BM * BK) % (4 * NUM_THREADS) != 0"
            continue
          fi
          # (BN * BK) must be divisible by (4 * NUM_THREADS) for vectorized B load loop
          if [[ $(( ($bn * $bk) % (4 * $NUM_THREADS) )) -ne 0 ]]; then
            echo "Skipping $config: (BN * BK) % (4 * NUM_THREADS) != 0"
            continue
          fi
          # TN must be >= 4 for float4 writeback
          if [[ $tn -lt 4 ]]; then
            echo "Skipping $config: TN < 4"
            continue
          fi

          # Update the parameters in runner.cu (inside runSgemmAutotuned)
          sed -i '/void runSgemmAutotuned/,/^}/ {
            s/const int BM = .*/const int BM = '"$bm"';/
            s/const int BN = .*/const int BN = '"$bn"';/
            s/const int BK = .*/const int BK = '"$bk"';/
            s/const int TM = .*/const int TM = '"$tm"';/
            s/const int TN = .*/const int TN = '"$tn"';/
          }' $RUNNER

          # Rebuild and run
          echo "($CONFIG_NUM/$TOTAL_CONFIGS): $config" | tee -a $OUTPUT
          make build 2>&1 | tail -1

          if [[ $? -ne 0 ]]; then
            echo "BUILD FAILED for $config" | tee -a $OUTPUT
            continue
          fi

          # Run benchmark, kill after 15 seconds if stuck
          timeout -v 15 build/sgemm_practice 9 2>&1 | tee -a $OUTPUT
        done
      done
    done
  done
done

# Restore default parameters
sed -i '/void runSgemmAutotuned/,/^}/ {
  s/const int BM = .*/const int BM = 128;/
  s/const int BN = .*/const int BN = 128;/
  s/const int BK = .*/const int BK = 8;/
  s/const int TM = .*/const int TM = 8;/
  s/const int TN = .*/const int TN = 8;/
}' $RUNNER

echo ""
echo "Results saved to $OUTPUT"
echo "Best configuration (by 4096 GFLOPS):"
best_line=$(grep "size: (4096)" $OUTPUT | sort -t'(' -k3 -rn | head -1)
grep -n "$best_line" $OUTPUT | head -1 | while IFS=: read -r lineno _; do
  sed -n "$((lineno-14)),$((lineno))p" $OUTPUT
done
