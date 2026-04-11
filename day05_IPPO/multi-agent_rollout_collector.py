import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#from pettingzoo.mpe import simple_spread_v3 # eski kütüphane
from mpe2 import simple_spread_v3

env = simple_spread_v3.parallel_env(N=3)
obs, infos = env.reset()
agents = env.agents

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.pi = nn.Linear(128, act_dim)    #actor
        self.v = nn.Linear(128, 1) #critic

    def forward(self, x):
        h = self.net(x)
        logits = self.pi(h)
        value = self.v(h)

        dist = torch.distributions.Categorical(logits=logits)
        return dist, value.squeeze(-1)

obs_dim = obs[agents[0]].shape[0]
act_dim = env.action_space(agents[0]).n  #action space = Discrete(5) buradaki 5'i çekmek için .n

policy = Policy(obs_dim, act_dim)  #policy network

# her agent için "obs","actions", "rewards", "dones", "log_probs" ve "values" değerleri return edilir
def init_buffer(agents):
    return {
        agent: {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": []
        }
        for agent in agents
    }

# rollout collector
def collect_rollout(env, policy, rollout_length=100):
    obs, _ = env.reset()
    agents = env.agents[:]  # kopyala

    buffer = init_buffer(agents)

    step = 0
    while step < rollout_length:

        # Episode bitti mi kontrol et
        if not env.agents:
            obs, _ = env.reset()
            agents = env.agents[:]
            continue

        actions = {}
        log_probs = {}
        values = {}

        for agent in env.agents:  # env.agents kullan, eski liste değil
            o = torch.tensor(obs[agent], dtype=torch.float32)
            dist, value = policy(o)
            action = dist.sample()

            actions[agent] = action.item()
            log_probs[agent] = dist.log_prob(action).item()
            values[agent] = value.item()

        next_obs, rewards, terms, truncs, infos = env.step(actions)

        for agent in env.agents:
            done = terms.get(agent, False) or truncs.get(agent, False)

            buffer[agent]["obs"].append(obs[agent])
            buffer[agent]["actions"].append(actions[agent])
            buffer[agent]["rewards"].append(rewards[agent])
            buffer[agent]["dones"].append(done)
            buffer[agent]["log_probs"].append(log_probs[agent])
            buffer[agent]["values"].append(values[agent])

        step += 1

        # Episode bittiyse reset
        episode_done = all(terms.get(a, False) or truncs.get(a, False)
                          for a in agents)
        if episode_done or not next_obs:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return buffer


if __name__ == "__main__":

    buffer = collect_rollout(env, policy, rollout_length=100)

    print("\n=== DEBUG ===")

    for agent in buffer:
        print(f"\nAgent: {agent}")
        print("Steps:", len(buffer[agent]["obs"]))
        print("Total reward:", sum(buffer[agent]["rewards"]))

    # action dağılımı
    from collections import Counter

    for agent in buffer:
        print(f"\nAction dist ({agent}):")
        print(Counter(buffer[agent]["actions"]))