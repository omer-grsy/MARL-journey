import numpy as np
import yaml
from mpe2 import simple_spread_v3


class FaultWrapper:
    def __init__(self, env, fault_config):
        self.env = env


        self.fault_type = fault_config.get("type", "none")
        self.faulty_agents = fault_config.get("agents", [])
        #self.drop_prob = fault_config.get("drop_prob", 0.5)
        # intermittent
        self.fault_period = fault_config.get("fault_period", 10)
        self.fault_duration = fault_config.get("fault_duration", 5)

        self.timestep = 0

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self):
        self.timestep = 0
        obs, infos = self.env.reset()
        obs = self._apply_obs_fault(obs)  #reset'te de uygula
        return obs, infos

    def step(self, actions):
        actions = self._apply_action_fault(actions)

        obs, rewards, terms, truncs, infos = self.env.step(actions)

        self.timestep += 1

        obs = self._apply_obs_fault(obs)

        return obs, rewards, terms, truncs, infos

    # -----------------------
    # ACTION FAULT
    # -----------------------
    def _apply_action_fault(self, actions):
        if self.fault_type == "none":
            return actions

        new_actions = actions.copy()

        for agent in actions:
            if agent not in self.faulty_agents:
                continue

            # FAIL-STOP → no-op
            if self.fault_type == "fail_stop":
                new_actions[agent] = 0

            # BYZANTINE → random
            elif self.fault_type == "byzantine":
                new_actions[agent] = self.env.action_space(agent).sample()

            # INTERMITTENT → random drop
            elif self.fault_type == "intermittent":
                if np.random.rand() < 0.5:
                    new_actions[agent] = 0

        return new_actions

    # -----------------------
    # OBS FAULT
    # -----------------------
    def _apply_obs_fault(self, obs):
        if self.fault_type == "none":
            return obs

        # -------- FAIL STOP --------
        if self.fault_type == "fail_stop":
            for agent in self.faulty_agents:
                if agent in obs:
                    obs[agent] = np.zeros_like(obs[agent])
            return obs

        # -------- BYZANTINE (FDI) --------
        elif self.fault_type == "byzantine":
            agents = self.possible_agents
            n_agents = len(agents)

            for faulty in self.faulty_agents:
                faulty_idx = agents.index(faulty)

                for agent in agents:
                    if agent == faulty or agent not in obs:
                        continue

                    other_indices = [k for k in range(n_agents) if k != agents.index(agent)]

                    if faulty_idx in other_indices:
                        j = other_indices.index(faulty_idx)

                        start = 10 + j * 2
                        end = start + 2
                        fake_pos =  obs[agent][start:end]
                        obs[agent][start:end] = -2 * fake_pos
                        # obs[agent][start:end] = np.random.uniform(
                        #     -2.0, 2.0, 2
                        # ).astype(np.float32)

            return obs

        # -------- INTERMITTENT --------
        elif self.fault_type == "intermittent":
            for agent in self.faulty_agents:
                if agent in obs and np.random.rand() < 0.5: #if agent in obs and np.random.rand() < self.drop_prob:
                    obs[agent] = np.zeros_like(obs[agent])
            return obs

        return obs


# =========================
# CONFIG
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

env = simple_spread_v3.parallel_env(N=3)

FAULT_TYPE = "byzantine"

env = FaultWrapper(env, cfg["faults"][FAULT_TYPE])

#print(cfg["faults"][FAULT_TYPE])
# =========================
# TEST
# =========================
obs, _ = env.reset()

for step in range(5):
    actions = {a: env.env.action_space(a).sample() for a in env.env.agents}

    obs, rewards, terms, truncs, infos = env.step(actions)

    print(f"\nSTEP {step}")
    print("agent_0 obs first 4:", obs["agent_0"][:4])