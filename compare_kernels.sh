#!/bin/bash

# Compare performance between original and practice kernels
# Usage:
#   ./compare_kernels.sh 1              # Compare kernel 1 (original vs practice)
#   ./compare_kernels.sh 1 --practice   # Only run practice kernel 1
#   ./compare_kernels.sh 1 2 3          # Compare kernels 1, 2, 3

BUILD_DIR="build"
PRACTICE_ONLY=false
KERNELS=()

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--practice" ]] || [[ "$arg" == "-p" ]]; then
        PRACTICE_ONLY=true
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        KERNELS+=("$arg")
    else
        echo "Unknown argument: $arg"
        echo "Usage: $0 <kernel_numbers...> [--practice|-p]"
        exit 1
    fi
done

# Check if kernel numbers provided
if [ ${#KERNELS[@]} -eq 0 ]; then
    echo "Error: No kernel numbers provided"
    echo "Usage: $0 <kernel_numbers...> [--practice|-p]"
    echo "Example: $0 1 2 3"
    echo "Example: $0 1 --practice  (only run practice kernel)"
    exit 1
fi

# Build both versions
echo "Building kernels..."
make build > /dev/null 2>&1

# Function to extract GFLOPS from output
extract_gflops() {
    local output="$1"
    echo "$output" | grep "performance:" | awk '{print $4}' | tr -d '()'
}

# Function to run kernel and get results
run_kernel() {
    local executable="$1"
    local kernel_num="$2"
    local label="$3"

    echo ""
    echo "=== $label - Kernel $kernel_num ==="

    if [ ! -f "$BUILD_DIR/$executable" ]; then
        echo "Error: $BUILD_DIR/$executable not found"
        return 1
    fi

    output=$($BUILD_DIR/$executable $kernel_num 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "❌ FAILED (exit code: $exit_code)"
        echo "$output" | tail -10
        return 1
    fi

    # Check for verification failure
    if echo "$output" | grep -q "Failed to pass"; then
        echo "❌ VERIFICATION FAILED"
        echo ""
        echo "Error details:"
        echo "$output" | grep -A 5 "Failed to pass"
        return 1
    fi

    echo "✓ PASSED"
    echo ""
    echo "Performance (GFLOPS) for each size:"
    echo "------------------------------------"

    # Extract all GFLOPS values with sizes
    echo "$output" | grep "performance:" | while read -r line; do
        size=$(echo "$line" | sed -n 's/.*size: (\([0-9]*\)).*/\1/p')
        gflops=$(echo "$line" | sed -n 's/.*performance: ( *\([0-9.]*\)).*/\1/p')
        printf "  Size %4s: %8.1f GFLOPS\n" "$size" "$gflops"
    done

    return 0
}

# Compare kernels
for kernel in "${KERNELS[@]}"; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Comparing Kernel $kernel"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Run original kernel (unless practice-only mode)
    if [ "$PRACTICE_ONLY" = false ]; then
        run_kernel "sgemm" "$kernel" "ORIGINAL"
        original_status=$?
    fi

    # Run practice kernel
    run_kernel "sgemm_practice" "$kernel" "PRACTICE"
    practice_status=$?

    # Summary
    echo ""
    if [ "$PRACTICE_ONLY" = false ]; then
        if [ $original_status -eq 0 ] && [ $practice_status -eq 0 ]; then
            echo "✓ Both versions passed"
        elif [ $practice_status -eq 0 ]; then
            echo "⚠ Original failed, Practice passed"
        elif [ $original_status -eq 0 ]; then
            echo "⚠ Original passed, Practice failed"
        else
            echo "❌ Both versions failed"
        fi
    else
        if [ $practice_status -eq 0 ]; then
            echo "✓ Practice version passed"
        else
            echo "❌ Practice version failed"
        fi
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Comparison complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Always exit with success so make doesn't fail
exit 0
