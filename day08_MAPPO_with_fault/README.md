# MARL Onboarding - Day 8: MAPPO with Commnet and Fault Injection
This project implements a simple FaultWrapper for injecting faults into the simple_spread_v3 multi-agent environment.
The environment is wrapped using the FaultWrapper class, so there is no need to modify the original environment.
3 fault types are implemented which are fail stop, byzantine and intermittent.
When fail stop fault occurs, the agent's observation and action are set to 0.
When byzantine fault occurs, other agents receive corrupted position info about the faulty agent.
When intermittent fault occurs, agent randomly drops out as if it has fail stop fault.
Four fault scenarios are investigated: no fault, a single agent subjected to a fail-stop fault, a single agent subjected to a Byzantine fault, and two agents subjected to intermittent faults.
Each scenario is evaluated using three different random seeds (0, 1, and 2) to ensure robustness to stochasticity.
In total, 12 training runs are conducted.

## Setup
```bash
pip install -r requirements.txt
```
## Usage
```bash
bash run_all.sh
```
## Configurations
The `configs/` directory contains YAML files defining different fault scenarios (no fault, fail-stop, byzantine, intermittent) used in the experiments.

## Metrics
- Coperative reward: Average shared reward received by all agents at each timestep
- Coverage ratio : The fraction of landmarks that are within a predefined distance threshold of at least one agent.
- Average distance : The average of the minimum distances between each landmark and the closest agent
- Policy entropy : Measures the randomness of agent policies. Higher entropy indicates more exploration.
- Message entropy(approximate): Estimated entropy of communication messages derived from hidden states of agents.
- Value loss : Mean squared error between the predicted state value and the computed return.
- Policy loss : Clipped surrogate loss used to update the policy in PPO:

## Goal
The goal of this project is to analyze how multi-agent systems with communication behave under different fault conditions, focusing on robustness and coordination performance.

## Logging
Experiments are logged using Weights & Biases (wandb).