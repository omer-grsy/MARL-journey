from mpe2 import simple_spread_v3
import numpy as np

env = simple_spread_v3.parallel_env(N=3) # default N=3, eğer arttırmak istersen () içine N=5 or N=7
obs, infos = env.reset()
print(env.agents)
print(obs.keys())

agent = env.agents[0]
print(env.observation_space(agent))


print(obs["agent_0"])
print(len(obs["agent_0"]))
print(env.action_space("agent_0"))

for _ in range(5):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

    print("Rewards:", rewards)

# o = obs["agent_0"] ---> o[0:2] ----> self velocity
                   #      o[2:4] ----> self position
                   #      o[4:6] ----> landmark_0 - self_position (relative position)
                   #      o[6:8] ----> landmark_1 - self_position
                   #      o[8:10] ----> landmark_2 - self_position
                   #      o[10:12] ----> agent_1 - self_position
                   #      o[12:14] ----> agent_2 - self_position
                   #      o[14:17] ----> communication padding(simple spread'de genelde kullanılmaz)
# Note: Diğer ajanların hızı bilinmez çünkü bu sistem partially observable
# Hız da bilinseydi zaten problem çok kolay olurdu.
# Reward shared kullanılır yani global reward. Ya hepimiz kazanırız ya hepimiz kaybederiz mantığı
# Cooperation'a teşvik
# reward = - landmark_distance - collision_penalty

# action space ---> 5 tane ---> 0 hiçbir şey yapma(no-op)
                             #  1 sola
                             #  2 sağa
                             #  3 aşağı
                             #  4 yukarı
# Agent 2D kuvvet vektörü üretiyor(note that position is also 2D)
# sola gitmek için Force ---> (-1,0)
# sağa gitmek için Force ---> (1,0)
# aşağı gitmek için Force ---> (0,-1)
# yukarı gitmek için Force ---> (0,1)

# her stepde aşağıdakiler güncellenir
#velocity += force
#velocity *= damping
#position += velocity
