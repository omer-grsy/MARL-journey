# render_ippo.py
import numpy as np
import torch
import torch.nn as nn
import imageio
from mpe2 import simple_spread_v3

# =========================
# POLICY (aynı mimari)
# =========================
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.pi = nn.Linear(128, act_dim)
        self.v  = nn.Linear(128, 1)

    def forward(self, x):
        h     = self.net(x)
        dist  = torch.distributions.Categorical(logits=self.pi(h))
        value = self.v(h).squeeze(-1)
        return dist, value


# =========================
# CHECKPOINT YÜKLE
# =========================
# Önce eğitim koduna şunu eklemen lazım:
# torch.save(policy.state_dict(), "ippo_policy.pth")
# Eğitim bittikten sonra bu dosya oluşacak

device  = torch.device("cpu")  # render için CPU yeterli
env = simple_spread_v3.parallel_env(N=3, max_cycles=100,
                                    render_mode="rgb_array",
                                    dynamic_rescaling=False,
                                    continuous_actions=False)
obs, _  = env.reset()
agents  = env.possible_agents

obs_dim = env.observation_space(agents[0]).shape[0]
act_dim = env.action_space(agents[0]).n

policy = Policy(obs_dim, act_dim).to(device)
policy.load_state_dict(torch.load("ippo_policy.pth", map_location=device))
policy.eval()

# =========================
# GIF KAYDET
# =========================
frames = []
obs, _ = env.reset()

for _ in range(100):
    if not env.agents:
        break

    actions = {}
    for agent in env.agents:
        o          = torch.tensor(obs[agent], dtype=torch.float32)
        with torch.no_grad():
            dist, _ = policy(o)
        actions[agent] = dist.sample().item()

    obs, _, terms, truncs, _ = env.step(actions)
    for entity in env.unwrapped.world.landmarks:
        print(f"Landmark absolute pos: {entity.state.p_pos}")
    #print(f"Frame {_}: landmarks = {obs[agents[0]][4:10]}")
    frame = env.render()
    frames.append(frame)

env.close()
print(f"Toplam frame: {len(frames)}")
imageio.mimsave("ippo_n3.gif", frames, fps=15)
print("GIF kaydedildi: ippo_n3.gif")