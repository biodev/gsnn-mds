#!/bin/bash

# =============================================================================
# Drug Response Prediction Pipeline Runner
# =============================================================================

set -e  # Exit on any error

# Default values
CONFIG="config/config.yaml"
CORES=8
JOBS=1
DRYRUN=false
TARGET=""
CLUSTER=""

# Help function
show_help() {
    cat << EOF
Drug Response Prediction Pipeline Runner

Usage: $0 [OPTIONS]

Options:
    -c, --config FILE       Configuration file (default: config/config.yaml)
    -j, --cores N          Number of cores to use (default: 8)
    -J, --jobs N           Number of parallel jobs for cluster (default: 1)
    -t, --target TARGET    Specific target to build (optional)
    -d, --dry-run          Perform dry run without execution
    -C, --cluster CMD      Cluster submission command template
    -h, --help             Show this help message

Examples:
    # Run complete pipeline with default config
    $0

    # Run test configuration
    $0 --config config/config_test.yaml --cores 4

    # Dry run to see what will be executed
    $0 --dry-run

    # Run only graph construction
    $0 --target exp/main_experiment/graph/graph.pt

    # Run on SLURM cluster
    $0 --cluster "sbatch --cpus-per-task={threads} --mem={resources.mem_mb}M --time={resources.runtime}" --jobs 5

    # Quick test run
    $0 --config config/config_test.yaml --cores 2 --dry-run

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -j|--cores)
            CORES="$2"
            shift 2
            ;;
        -J|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRYRUN=true
            shift
            ;;
        -C|--cluster)
            CLUSTER="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file '$CONFIG' not found!"
    echo "Available configs:"
    ls -la config/*.yaml 2>/dev/null || echo "  No config files found in config/ directory"
    exit 1
fi

# Build snakemake command
SNAKE_CMD="snakemake --snakefile workflow/Snakefile --configfile $CONFIG"

# Add cores/jobs
if [[ -n "$CLUSTER" ]]; then
    SNAKE_CMD="$SNAKE_CMD --cluster \"$CLUSTER\" --jobs $JOBS"
else
    SNAKE_CMD="$SNAKE_CMD --cores $CORES"
fi

# Add target if specified
if [[ -n "$TARGET" ]]; then
    SNAKE_CMD="$SNAKE_CMD $TARGET"
fi

# Add dry run if requested
if [[ "$DRYRUN" == "true" ]]; then
    SNAKE_CMD="$SNAKE_CMD --dry-run"
fi

# Print configuration info
echo "=== Drug Response Prediction Pipeline ==="
echo "Config: $CONFIG"
echo "Cores: $CORES"
if [[ -n "$CLUSTER" ]]; then
    echo "Cluster: $CLUSTER"
    echo "Jobs: $JOBS"
fi
if [[ -n "$TARGET" ]]; then
    echo "Target: $TARGET"
fi
if [[ "$DRYRUN" == "true" ]]; then
    echo "Mode: DRY RUN"
fi
echo "=========================================="
echo

# Execute command
echo "Running: $SNAKE_CMD"
echo
eval $SNAKE_CMD 