#!/bin/bash
# 4 senaryo x 3 seed = 12 eğitim
# Kullanım: bash run_all.sh

FAULT_TYPES=("none" "fail_stop" "byzantine" "intermittent")
SEEDS=(0 1 2)

for fault in "${FAULT_TYPES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=============================="
        echo "Fault: $fault | Seed: $seed"
        echo "=============================="
        python mappo_train.py --fault_type "$fault" --seed "$seed"
    done
done

echo "Tüm eğitimler tamamlandı."