#!/usr/bin/env bash
# Day 9 full experiment sweep.
#
# Matrix: 3 strategies x (S1 no-fault + S2 fail_stop + S3 byzantine + S4 intermittent)
#         x 3 seeds = 36 runs
#
# Usage:
#   bash run_all.sh            # full-length runs (TOTAL_UPDATES=3000)
#   bash run_all.sh smoke      # abbreviated smoke runs (TOTAL_UPDATES=150)

set -euo pipefail

MODE="${1:-full}"
EXTRA=""
if [[ "$MODE" == "smoke" ]]; then
    EXTRA="--total_updates 150 --rollout_steps 500"
    echo "[run_all_day9] SMOKE mode: $EXTRA"
fi

SCENARIOS=(S1_nofault S2_fail_stop S3_byzantine S4_intermittent)
STRATEGIES=(A B C)
SEEDS=(0 1 2)

for strat in "${STRATEGIES[@]}"; do
    for sc in "${SCENARIOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "=== strategy=$strat scenario=$sc seed=$seed ==="
            python3 model_train.py \
                --config "configs/${sc}.yaml" \
                --strategy "$strat" \
                --seed "$seed" \
                --topology "configs/topology_full.json" \
                --wandb \
                $EXTRA
        done
    done
done
