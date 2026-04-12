from mpe2 import simple_spread_v3
import numpy as np

for N in [3, 5]:
    print(f"\n{'='*40}")
    print(f"N = {N} AJAN")
    print(f"{'='*40}")

    env = simple_spread_v3.parallel_env(N=N)
    obs, _ = env.reset()
    agents = env.possible_agents

    agent = agents[0]
    obs_space = env.observation_space(agent)
    act_space = env.action_space(agent)

    print(f"\nAgent: {agent}")
    print(f"Obs space : {obs_space}")
    print(f"Obs dim   : {obs_space.shape[0]}")
    print(f"Act space : {act_space}")
    print(f"Act dim   : {act_space.n}")

    print(f"\nAction anlamları:")
    print(f"  0 → no_action")
    print(f"  1 → left")
    print(f"  2 → right")
    print(f"  3 → down")
    print(f"  4 → up")

    # Obs boyutları
    vel_end        = 2
    pos_end        = 4
    landmark_end   = pos_end + 2 * N
    other_ag_end   = landmark_end + 2 * (N - 1)
    comm_end       = obs_space.shape[0]
    comm_dim       = comm_end - other_ag_end

    print(f"\nObs içeriği (N={N}, toplam={obs_space.shape[0]}):")
    print(f"  [0:2]              → kendi hızı (vx, vy)")
    print(f"  [2:4]              → kendi pozisyonu (x, y)")
    print(f"  [4:{landmark_end}]"
          f"{'':>10} → landmark rölatif pozisyonları ({N} landmark × 2 = {2*N})")
    print(f"  [{landmark_end}:{other_ag_end}]"
          f"{'':>8} → diğer ajanların rölatif pozisyonları ({N-1} ajan × 2 = {2*(N-1)})")
    print(f"  [{other_ag_end}:{comm_end}]"
          f"{'':>8} → communication channel ({comm_dim} değer)")
    print(f"\n  NOT: IPPO'da communication channel sıfır —")
    print(f"       ajanlar mesaj göndermiyor.")
    print(f"       Gün 6'da CommNet ile bu kanal aktif hale gelecek.")

    # Gerçek obs değerlerine bak
    real_obs = obs[agent]
    print(f"\nİlk reset sonrası örnek obs (agent_0):")
    print(f"  Hız          : {real_obs[0:2]}")
    print(f"  Pozisyon     : {real_obs[2:4]}")
    print(f"  Landmarks    : {real_obs[4:landmark_end]}")
    print(f"  Diğer ajanlar: {real_obs[landmark_end:other_ag_end]}")
    print(f"  Comm channel : {real_obs[other_ag_end:comm_end]}")

    env.close()