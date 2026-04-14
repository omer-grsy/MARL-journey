# MARL Onboarding - Day 7: Fault Injection Framework
This project implements a simple FaultWrapper for injecting faults into the simple_spread_v3 multi-agent environment.
The environment is wrapped using the FaultWrapper class, so there is no need to modify the original environment.
3 fault types are implemented which are fail stop, byzantine and intermittent.
When fail stop fault occurs, the agent's observation and action are set to 0.
When byzantine fault occurs, other agents receive corrupted position info about the faulty agent.
When intermittent fault occurs, agent randomly drops out as if it has fail stop fault.

## Setup
```bash
pip install -r requirements.txt
```
## Usage
Modify FAULT_TYPE variable in fault_wrapper.py and run it
## Goal
Test how multi-agent systems behave under specific faults
