# render.py
import os
import numpy as np
import torch
import torch.nn as nn
import imageio
from mpe2 import simple_spread_v3

# =========================
# AYARLAR
# =========================
N_AGENTS   = 3
HIDDEN_DIM = 128
CKPT_PATH  = "checkpoints/model_intermittent_seed2.pt"

# GIF ismi checkpoint'ten otomatik türetiliyor:
#   checkpoints/model_none_seed0.pt  ->  renders/model_none_seed0.gif
GIF_DIR  = "renders"
GIF_PATH = os.path.join(
    GIF_DIR,
    os.path.splitext(os.path.basename(CKPT_PATH))[0] + ".gif"
)
os.makedirs(GIF_DIR, exist_ok=True)

# =========================
# ACTOR (eğitimdekiyle aynı)
# =========================
class CommNetActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, act_dim)
        )

    def forward(self, obs_agents, adj):
        h   = self.encoder(obs_agents)
        msg = torch.einsum("ij,jbd->ibd", adj, h)
        out = torch.cat([h, msg], dim=-1)
        return self.policy(out)


# =========================
# ENV
# =========================
device = torch.device("cpu")
env = simple_spread_v3.parallel_env(N=N_AGENTS, max_cycles=100,
                                    render_mode="rgb_array",
                                    dynamic_rescaling=False,
                                    continuous_actions=False)
obs, _  = env.reset()
agents  = env.possible_agents
obs_dim = env.observation_space(agents[0]).shape[0]
act_dim = env.action_space(agents[0]).n

# =========================
# CHECKPOINT YÜKLE
# =========================
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
actor = CommNetActor(obs_dim, act_dim).to(device)
actor.load_state_dict(ckpt["actor"])
actor.eval()

obs_mean = np.asarray(ckpt["obs_mean"])
obs_var  = np.asarray(ckpt["obs_var"])

# Full-connected normalized adjacency (eğitimdekiyle aynı)
adj = torch.ones(N_AGENTS, N_AGENTS, device=device)
adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1.0)

# =========================
# GIF KAYDET
# =========================
frames = []
obs, _ = env.reset()

for t in range(100):
    if not env.agents:
        break

    # Ajan obs'larını sırayla al, normalize et, stack'le
    obs_list   = [obs[a] for a in agents]
    obs_norm   = [(o - obs_mean) / (np.sqrt(obs_var) + 1e-8) for o in obs_list]
    obs_tensor = torch.from_numpy(
        np.array(obs_norm, dtype=np.float32)
    ).unsqueeze(1).to(device)   # (n_agents, 1, obs_dim)

    with torch.no_grad():
        logits = actor(obs_tensor, adj)   # (n_agents, 1, act_dim)

    actions = {}
    for i, a in enumerate(agents):
        dist = torch.distributions.Categorical(logits=logits[i, 0])
        actions[a] = dist.sample().item()

    obs, _, _, _, _ = env.step(actions)
    frames.append(env.render())

env.close()
print(f"Toplam frame: {len(frames)}")
imageio.mimsave(GIF_PATH, frames, fps=15)
print(f"GIF kaydedildi: {GIF_PATH}")