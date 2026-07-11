from __future__ import annotations
import gymnasium as gym
from vizdoom import gymnasium_wrapper  # noqa: F401  (必须导入注册环境)
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
def preprocess_obs(obs):
    screen = obs["screen"]  # (240, 320, 3)

    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) # 这里是灰度化，不是通道数减少
    screen = cv2.resize(screen, (84, 84))
    screen = screen.astype(np.float32) / 255.0
    return screen

class ReplayBuffer:
    def __init__(self, capacity=100000):
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

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x: (B, 84, 84)
        x = x.unsqueeze(1)  # -> (B, 1, 84, 84)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def plot_curves(reward_history, loss_history):
    plt.figure()

    # reward curve
    plt.subplot(2, 1, 1)
    plt.plot(reward_history)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # loss curve
    plt.subplot(2, 1, 2)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig("DQN-VizdoomBasic-v1_training.png")
    plt.show()

def train(
    env_id="VizdoomBasic-v1",
    episodes=5000,
    gamma=0.995,
    lr=1e-4,
    batch_size=32,
    buffer_size=100000,
    target_update=100,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=50000,
    device="cuda" if torch.cuda.is_available() else "cpu"
):

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

        while True:
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
        reward_history.append(total_reward)
        loss_history.append(loss_value)
        print(f"Episode {ep}, reward={total_reward:.2f},loss={loss_value:.4f}")

    env.close()

    return policy_net, reward_history, loss_history

def test(env_id="VizdoomBasic-v1", model=None, c=3, trial=3, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = gym.make(env_id, render_mode="human")

    obs, _ = env.reset()
    state = preprocess_obs(obs)

    for ep in range(trial):

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

        print(f"[TEST]: reward={total_reward:.2f}")
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


if __name__ == "__main__":
    #play_env("VizdoomBasic-v1")
    model, reward_history, loss_history = train()
    plot_curves(reward_history, loss_history)
    os.system("pause")
    test(model=model,trial=3)