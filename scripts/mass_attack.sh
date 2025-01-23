#!/bin/bash

###############################################################################
# Usage:
#   bash run_parallel_attacks.sh /path/to/config.json
#
# Description:
#   - Iterates over attack_idx = 0..49.
#   - Launches each run on any GPU that is currently free (no processes).
#   - Runs python -m mu_attack.execs.attack with the given --config_path 
#     plus the --attack_idx override.
#   - Waits for all jobs to finish at the end.
###############################################################################

# 1) Check required argument: config path
if [ -z "$1" ]; then
  echo "Usage: bash $0 /path/to/config.json"
  exit 1
fi

CONFIG_PATH="$1"

# 2) Define the range of attack indices
START_IDX=0
END_IDX=40

# 3) Define GPU range
GPU_START=0
GPU_END=7

# 4) A helper function to check if a GPU is free.
#    We'll say a GPU is "free" if it has zero processes running.
check_gpu() {
    local GPU_ID=$1
    # Query how many PIDs (processes) are on that GPU
    local num_procs
    num_procs=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader | wc -l)
    echo "$num_procs"
}

echo "================================================================="
echo "Config path: $CONFIG_PATH"
echo "attack_idx range: $START_IDX..$END_IDX"
echo "GPU range: $GPU_START..$GPU_END"
echo "================================================================="

# 5) Main loop over attack_idx
for idx in $(seq "$START_IDX" "$END_IDX"); do

    # 5a) Find an available GPU
    GPU=-1
    while [[ $GPU -lt $GPU_START ]]; do
        # Check each GPU in the range
        for (( i=GPU_START; i<=GPU_END; i++ )); do
            if [[ $(check_gpu "$i") -eq 0 ]]; then
                GPU=$i
                break
            fi
        done

        # If all GPUs are busy, wait a bit and check again
        if [[ $GPU -lt $GPU_START ]]; then
            echo "All GPUs busy; waiting..."
            sleep 5
        fi
    done

    # 5b) Launch the job on the found GPU in the background
    echo "Launching attack_idx=$idx on GPU $GPU"
    CUDA_VISIBLE_DEVICES="$GPU" python -m mu_attack.execs.attack \
        --config_path "$CONFIG_PATH" \
        --attack_idx "$idx" &

    # Optional: Sleep briefly so we don't check GPU availability too rapidly
    sleep 3

done

# 6) Wait for all background jobs to complete before exiting
wait
echo "All jobs completed!"
