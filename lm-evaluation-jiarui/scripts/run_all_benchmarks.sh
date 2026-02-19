#!/bin/bash
# Run all proxy benchmarks for all models
# This is the master script that runs Oolong, BigCodeBench, and SciCode
#
# Usage: ./run_all_benchmarks.sh [--limit N] [--benchmark oolong|bigcodebench|scicode|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
LIMIT=""
BENCHMARK="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Proxy Benchmark Suite"
echo "============================================"
echo "Benchmark: ${BENCHMARK}"
echo "Limit: ${LIMIT:-none}"
echo ""

run_oolong() {
    echo "============================================"
    echo "Running Oolong Benchmark"
    echo "============================================"

    # Run synth dataset
    echo "--- Oolong Synth ---"
    bash "${SCRIPT_DIR}/run_oolong_all.sh" --dataset synth $LIMIT

    # Run real dataset
    echo "--- Oolong Real ---"
    bash "${SCRIPT_DIR}/run_oolong_all.sh" --dataset real $LIMIT
}

run_bigcodebench() {
    echo "============================================"
    echo "Running BigCodeBench Benchmark"
    echo "============================================"

    # Run instruct split
    echo "--- BigCodeBench Instruct ---"
    bash "${SCRIPT_DIR}/run_bigcodebench_all.sh" --split instruct --subset full $LIMIT

    # Run complete split (optional, uncomment if needed)
    # echo "--- BigCodeBench Complete ---"
    # bash "${SCRIPT_DIR}/run_bigcodebench_all.sh" --split complete --subset full $LIMIT
}

run_scicode() {
    echo "============================================"
    echo "Running SciCode Benchmark"
    echo "============================================"

    # Run without background
    echo "--- SciCode Without Background ---"
    bash "${SCRIPT_DIR}/run_scicode_all.sh" --split test $LIMIT

    # Run with background
    echo "--- SciCode With Background ---"
    bash "${SCRIPT_DIR}/run_scicode_all.sh" --split test --with_background $LIMIT
}

case $BENCHMARK in
    oolong)
        run_oolong
        ;;
    bigcodebench)
        run_bigcodebench
        ;;
    scicode)
        run_scicode
        ;;
    all)
        run_oolong
        run_bigcodebench
        run_scicode
        ;;
    *)
        echo "Unknown benchmark: $BENCHMARK"
        echo "Valid options: oolong, bigcodebench, scicode, all"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "All requested benchmarks completed!"
echo "============================================"
