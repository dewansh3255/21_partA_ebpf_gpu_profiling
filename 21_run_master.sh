#!/bin/bash
# 21_run_master.sh
# Group 21 - GRS Project Part A (Distributed)

mkdir -p results/multi_node

echo "============================================================"
echo "  STARTING MASTER NODE (Rank 0)"
echo "  Waiting for Worker Node to connect..."
echo "============================================================"

# Using your laptop's static IP (192.168.1.1)
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=1 \
    --master_addr="192.168.52.110" \
    --master_port=29500 \
    21_ml_workload.py --gpus 1 --epochs 5 --output "results/multi_node/21_master_results.json"
