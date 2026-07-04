from __future__ import annotations
import gymnasium as gym
from vizdoom import gymnasium_wrapper  # noqa: F401  (必须导入注册环境)
import os
import random
import numpy as np

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
def preprocess_obs(obs):
    screen = obs["screen"]  # (240, 320, 3) RGB
    
    screen = cv2.resize(screen, (60, 45))  # 注意：OpenCV是(width, height)
    screen = screen.astype(np.float32) / 255.0# 归一化到 [0, 1]
    # 维度调整：从 (H, W, C) 变为 (C, H, W) 供PyTorch使用
    screen = np.transpose(screen, (2, 0, 1))  # -> (3, 45, 60)
    return screen

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, s, a, r, s2, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)
    
class CNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        
        # 2层卷积 + 2层池化
        self.cnn = nn.Sequential(
            # 第一层：32个 7×7 卷积核
            nn.Conv2d(3, 32, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 2×2 最大池化
            
            # 第二层：32个 4×4 卷积核
            nn.Conv2d(32, 32, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 2×2 最大池化
        )
        
        # 计算经过卷积和池化后的特征图尺寸
        # 输入：3 × 45 × 60
        # Conv1 (7×7, 无padding): 45→39, 60→54
        # Pool1 (2×2): 39→19, 54→27
        # Conv2 (4×4, 无padding): 19→16, 27→24
        # Pool2 (2×2): 16→8, 24→12
        # 输出：32 × 8 × 12 = 3072
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 12, 800), 
            nn.LeakyReLU(),               #默认alpha=0.01
            nn.Linear(800, n_actions)
        )
    
    def forward(self, x):
        # x: (B, 3, 45, 60)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def plot_curves(reward_history, loss_history, env_id):
    plt.figure()

    # reward curve
    plt.subplot(2, 1, 1)
    plt.plot(reward_history)
    plt.title(f"DQN-{env_id} Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # loss curve
    plt.subplot(2, 1, 2)
    plt.plot(loss_history)
    plt.title(f"DQN-{env_id}-version-65x45 Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(f"DQN-{env_id}-version-65x45 training.png")
    plt.show()

def train(
    env_id="VizdoomBasic-v1",
    episodes=3000,
    gamma=0.99,
    lr=0.01,
    batch_size=40,
    buffer_size=10000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=10000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    skipcount=5,
    target_update=10
):
    print(f"Training {env_id}...")

    env = gym.make(env_id)
    obs, _ = env.reset()

    obs_dim = preprocess_obs(obs).shape[0]
    n_actions = env.action_space.n

    policy_net = CNN(n_actions).to(device)
    target_net = CNN(n_actions).to(device)

    reward_history = []
    loss_history = []

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)

    step = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        state = preprocess_obs(obs)

        total_reward = 0
        loss_value = 0.0

        done = False

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-step / epsilon_decay)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(s).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = preprocess_obs(next_obs)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step += 1

            if len(buffer) > batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                    target = r + gamma * (1 - d) * q_next

                loss = F.mse_loss(q, target)
                #loss = F.smooth_l1_loss(q, target, beta=5)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()

            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break

            for i in range(skipcount):
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if i == skipcount-1:
                    state = preprocess_obs(next_obs)
                done = terminated or truncated
                if done:
                    break

        reward_history.append(total_reward)
        loss_history.append(loss_value)
        print(f"Episode {ep}, reward={total_reward:.2f},loss={loss_value:.4f}")

    env.close()

    return policy_net, reward_history, loss_history

def test(env_id="VizdoomDeadlyCorridor-v1", model=None, c=3, trial=3, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = gym.make(env_id, render_mode="human")

    for ep in range(trial):
        obs, _ = env.reset()
        state = preprocess_obs(obs)

        total_reward = 0

        while True:
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = model(s).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = preprocess_obs(next_obs)
            total_reward += reward
            print(f"reward:{reward}")
            os.system("pause")

            if done:
                break

        print(f"[TEST] Episode {ep}: reward={total_reward:.2f}")
        os.system("pause")
    env.close()

def play_env(env_id):
    env = gym.make(env_id, render_mode="human")
    for env_id in gym.envs.registry.keys():
        print(env_id)
    print(f"Running environment: {env_id}")
    print('env.spec:',env.spec)
    print('observation_space:',env.observation_space)
    print('action_space:',env.action_space)

    obs, info = env.reset()

    try:
        while True:
            #action = env.action_space.sample()
            action=int(input("请输入动作: "))#0 -none,1 -shoot,2-right 3-left

            obs, reward, terminated, truncated, info = env.step(action)
            print("状态:",obs)
            print("奖励:",reward)

            if terminated or truncated:
                os.system("pause")
                obs, info = env.reset()
                break


    except KeyboardInterrupt:
        print("\nExiting:", env_id)

    env.close()

ENV_LIST1 = [
    "VizdoomHealthGatheringSupreme-v1",
"VizdoomTakeCover-v1",
"VizdoomPredictPosition-v1",
"VizdoomMyWayHome-v1",
"VizdoomHealthGathering-v1",
"VizdoomDefendLine-v1",
"VizdoomDefendCenter-v1",
    "VizdoomDeadlyCorridor-v1"
]
ENV_LIST = [
    "VizdoomHealthGathering-v1"
]
if __name__ == "__main__":
    #play_env("VizdoomHealthGathering-v1")
    for env_id in ENV_LIST:
        model, reward_history, loss_history = train(env_id)
        plot_curves(reward_history, loss_history, env_id)
        os.system("pause")
        test(env_id, model=model,trial=3)
