# Day 9 — Fault-aware Training(Curriculum learning) + Topology Adaptation

MAPPO + CommNet üzerine fault-tolerant MARL. `simple_spread_v3` (mpe2 / PettingZoo), N=3 cooperative agent.

Üç strateji karşılaştırılmaktadır:
- **A** — naive baseline, fault'a duyarsız
- **B** — fault-aware curriculum + ground-truth fault indicator
- **C** — online detector + topology reconfiguration

## Kurulum

```bash
conda create -n rl-env python=3.10
conda activate rl-env
pip install -r requirements.txt
# ya da:
# pip install torch numpy pyyaml pettingzoo[mpe] mpe2 wandb
```

## Çalıştırma

Tek koşu:

```bash
python model_train.py --config configs/S3_byzantine.yaml --strategy C --seed 0 \
                --topology configs/topology_full.json
```

Tüm matris (3 strateji × 4 senaryo × 3 seed):

```bash
bash run_all.sh
```

Önemli flag'ler: `--total_updates`, `--rollout_steps` (smoke test için override), `--wandb` (W&B logging).

Çıktı: `checkpoints/day9_{strategy}_{scenario}_seed{seed}.pt`.

## Repo yapısı

- `train.py` - MAPPO + CommNet training loop, fault injection, detector entegrasyonu
- `src/adaptation.py` - `CurriculumScheduler`, `FaultDetector` (5 sinyal), `TopologyManager`
- `configs/` - senaryo YAML'leri ve topology JSON
- `run_all.sh` - full sweep
- `requirements.txt` - gerekli modüller

## Kısa mimari not

Actor (`CommNetActor`) iki adımda çalışır: `encode(obs)` → hidden state, sonra adjacency ile `aggregate_and_policy`. İkisi ayrı method, çünkü fault'u **mesaj seviyesinde** (encode sonrası, aggregation öncesi) enjekte etmek gerekiyor.

Detector beş sinyal üretir: `z_self`, `z_cv`, `z_fleet`, `cos_drop`, `coherence_drop`. Norm ve direction bazlı sinyallerin birleşimi sign-flip + random Byzantine'i de yakalıyor. Simetrik histerezi (set K=3, clear M=5) FP filtresi.

Strategy C tetiklenince `TopologyManager` faulty agent'ın in/out edge'lerini sıfırlar, self-loop kalır.

