import matplotlib.pyplot as plt
import random
import numpy as np
import gymnasium as gym
import os
#可调参数
gamma = 0.999  # Discount factor
num_iters = 1000  # Number of iterations
max_step=100 # 最大步数

custom_map = [
    "SFFFHFFFFHFFFFFF",
    "FHFFFFHFFFHFFFFF",
    "FFFHFFFFFHFFFHFF",
    "HFFFFHFFFFFFFHFF",
    "FFFFFHFFFHFFFFFF",
    "FFHFFFFFHFFFHFFF",
    "FFFFHFFFFFHFHFFF",
    "HFFFFFFHFFFFFHFH",
    "FFFFHFFFFFHFFFFF",
    "FFFHFFFFFHFHFFFF",
    "FFFFFHFFFFFHFHFF",
    "HFFFFFHFHFFFFFFF",
    "FFFFHFFFFFFHFFFF",
    "FFHFFFFFHFFFFHFF",
    "FFFFFHFFFFHFFFFF",
    "FHFFFFHFFFFFFFHG"
]#S: start, F: 冰面/路, H: hole, G: goal
# Now set up the environment
#env = gym.make('FrozenLake-v1',map_name="8x8")
env = gym.make('FrozenLake-v1',desc=custom_map,max_episode_steps=max_step)
print('env.spec:',env.spec)
print('observation_space:',env.observation_space)#格子编号 s = i * ncol + j
print('action_space:',env.action_space)#0: left, 1: down, 2: right, 3: up
def draw(env_desc, V, pi,num_iter=0):

    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    nrow, ncol = env_desc.shape

    k = num_iter
    value_grid = V[k].reshape(nrow, ncol)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    vmin, vmax = np.min(V), np.max(V)

    ax.imshow(value_grid, cmap='Blues', vmin=vmin, vmax=vmax)

    for i in range(nrow):
        for j in range(ncol):

            s = i * ncol + j

            tile = env_desc[i][j].decode() if hasattr(env_desc[i][j], "decode") else env_desc[i][j]

            # ✔ 终点/洞直接标记
            if tile in ['H', 'G', 'S']:
                ax.text(
                    j, i, tile,
                    ha='center', va='center',
                    color='red',
                    fontsize=14,
                    fontweight='bold'
                )
                continue

            # ✔ policy arrow（最后一轮）
            arrow = action_symbols[int(pi[k][s])]

            ax.text(
                j, i,
                f"{value_grid[i, j]:.2f}\n{arrow}",
                ha='center',
                va='center',
                fontsize=10,
                color='black'
            )

    ax.set_title(f"Final Policy (Iter {k+1})")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()
#值迭代：全局离线策略
def value_iteration(env, gamma, num_iters):
    env_desc = env.unwrapped.desc
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    mdp = env.unwrapped.P
    # P[s][a]=(p(s'|s,a), s', r, done)
    #print("map:",mdp)
    V  = np.zeros((num_iters + 1, num_states))#离散概率分布
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))#离散概率分布

    #开始收敛的轮次
    converge_iter = num_iters

    for k in range(1, num_iters + 1):
        #值迭代是全局离线策略，每次迭代都计算所有状态的所有动作的Q值
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[s][a]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..] 
                    pr = pxrds[0]  # p(s'\mid s,a)
                    nextstate = pxrds[1] # s'
                    reward = pxrds[2]
                    d=pxrds[3]#是否到达终点或洞
                    if d:
                        if reward>0:
                            #到达终点
                            reward=20
                        else:
                            #到达洞
                            reward=-20
                        #不加上未来奖励
                        Q[k,s,a] += pr *reward
                    else:
                        #reward=-10
                        Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])

            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])

        #只通过判断最后的策略结果是否改变来判断是否收敛
        if not np.array_equal(pi[k-1],pi[k]):
            converge_iter = k
    print("Final convergence iteration:", converge_iter)
    draw(env_desc, V[:-1], pi[:-1],converge_iter-1)
    return (pi,converge_iter)
def run(env, pi,num_iter):
    env_render = gym.make(
        "FrozenLake-v1",
        desc=custom_map,
        render_mode="human",
        is_slippery=True,
        max_episode_steps=max_step
    )#默认开启动作随机性
    #env_render = gym.make('FrozenLake-v1',map_name="8x8",render_mode="human")

    state, _ = env_render.reset(seed=0)
    done = False
    step=0
    win=False
    while not done:
        step+=1
        action = int(pi[num_iter][state])  # 用第num_iter轮策略
        state, reward, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        if reward==1 and terminated:
            win=True
    print(f"total_steps:{step},win:{win}")
    os.system("pause")
    env_render.close()
def play():
    env_play = gym.make(
        "FrozenLake-v1",
        desc=custom_map,
        render_mode="human",
        is_slippery=False
    )#游玩关闭随机性
    state, _ = env_play.reset(seed=0)
    done = False
    step=0
    win=False
    while not done:
        step+=1
        ch=input("请输入动作:A:左,S:下,D:右,W:上\n")
        ac=ch.upper()
        if ac=="A":
            action=0
        elif ac=="S":
            action=1
        elif ac=="D":
            action=2
        elif ac=="W":
            action=3
        else:
            print("输入错误")
            continue
        state, reward, terminated, truncated, _ = env_play.step(action)
        done = terminated or truncated
        if reward==1 and terminated:
            win=True
    print(f"total_steps:{step},win:{win}")
    os.system("pause")
    env_play.close()
#play()
pi,converge_iter = value_iteration(env=env, gamma=gamma, num_iters=num_iters)#返回训练之后的策略
os.system("pause")
run(env, pi, converge_iter)
