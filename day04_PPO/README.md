# 📘 Reinforcement Learning - PPO Experiments

This repository contains multiple implementations and experiments of Proximal Policy Optimization (PPO) and related reinforcement learning algorithms.

---

# 🚀 Implementations

## 🔹 Core Algorithms

- **Simple Actor-Critic**
  - Basic actor-critic architecture  
  - File: `Simple_Actor_Critic.py`

- **Simple PPO**
  - Minimal PPO implementation  
  - File: `SimplePPO.py`

---

## 🔹 PPO Variants

- **PPO with GAE**
  - Includes Generalized Advantage Estimation (GAE)  
  - File: `PPOwithGAE2.py`

- **PPO with Epochs & Minibatches**
  - Multi-epoch updates and minibatch training  
  - File: `PPO_with_epochs_minibatch.py`

- **CleanRL PPO**
  - Based on CleanRL implementation  
  - File: `CleanRL-PPO.py`

---

## 🔹 Hyperparameter Tuning

- **PPO + W&B Sweep**
  - Hyperparameter tuning with Weights & Biases  
  - File: `PPO_Hyperparameter_Sweep.py`

- **Sweep Configuration**
  - Search space definition  
  - File: `sweep.yaml`

---

# ⚙️ Features

- PPO from scratch  
- GAE (Generalized Advantage Estimation)  
- Minibatch + multi-epoch training  
- W&B experiment tracking  
- Bayesian hyperparameter optimization  

---

# 🧪 Running Experiments

## ▶️ Basic PPO

python SimplePPO.py

## ▶️ CleanRL PPO

python CleanRL-PPO.py

## ▶️ PPO Sweep

wandb sweep sweep.yaml  
wandb agent <ENTITY/PROJECT/SWEEP_ID>

---

# 📊 Tracking

Experiments are tracked with Weights & Biases (W&B):

- Episodic return  
- Training losses  
- KL divergence  
- Hyperparameters  

---

# 📈 Key Metrics

- Episodic Return  
- Policy Loss  
- Value Loss  
- Entropy  
- KL Divergence  

---

# 🎯 Goal

- Understand PPO step-by-step  
- Compare different PPO variants  
- Learn core RL techniques (GAE, clipping)  
- Optimize performance via hyperparameter sweeps  

---

# 🧠 Notes

- Sweep loop is handled by `wandb agent`  
- Each run uses a different hyperparameter configuration  
- Bayesian optimization improves search efficiency  