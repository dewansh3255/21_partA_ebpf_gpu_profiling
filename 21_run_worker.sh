#!/bin/bash
# 21_run_worker.sh
# Group 21 - GRS Project Part A (Distributed)

mkdir -p results/multi_node

echo "============================================================"
echo "  STARTING WORKER NODE (Rank 1)"
echo "  Connecting to Master Node at 192.168.1.1..."
echo "============================================================"

torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    21_ml_workload.py --gpus 1 --epochs 5 --output "results/multi_node/21_worker_results.json"

