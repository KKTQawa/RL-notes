from __future__ import annotations

import random
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

plt.rcParams["figure.figsize"] = (10, 5)

class Policy_Network(nn.Module):
    #这个更偏向于连续动作空间 Auxiliary Network
    #使用同一个网络来估计\mu和\sigma，然后根据\mu和\sigma固定的高斯分布来采样动作
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16  
        hidden_space2 = 32  

        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
    
class REINFORCE:

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)#把状态传入，获取参数\mu和\sigma

        #假设action符合高斯分布
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()

        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []#G_t

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        log_probs = torch.stack(self.probs).squeeze()

        # Update the loss with the mean log probability and deltas
        # Now, we compute the correct total loss by taking the sum of the element-wise products.
        loss = -torch.sum(log_probs * deltas)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
def save_checkpoint(agent, reward_over_episodes, episode=0, seed=5, path="REINFORCE_checkpoint.pth"):
    checkpoint = {
        "model_state_dict": agent.net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "reward_over_episodes": reward_over_episodes,
        "episode": episode,
        "seed": seed,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at episode {episode} → {path}")

def load_checkpoint(agent, path="REINFORCE_checkpoint.pth"):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return None, None

    checkpoint = torch.load(path)

    agent.net.load_state_dict(checkpoint["model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    reward_over_episodes = checkpoint["reward_over_episodes"]
    episode = checkpoint["episode"]
    seed = checkpoint["seed"]

    print(f"Checkpoint loaded from episode {episode} → {path}")
    return reward_over_episodes, episode, seed


# Create and wrap the environment
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(3e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

seed=5

# set seed
torch.manual_seed(seed)#控制动作采样
random.seed(seed)#控制环境随机性
np.random.seed(seed)

# Reinitialize agent every seed
agent = REINFORCE(obs_space_dims, action_space_dims)
reward_over_episodes = []

for episode in range(total_num_episodes):
    # gymnasium v26 requires users to set seed while resetting the environment
    obs, info = wrapped_env.reset(seed=seed)

    done = False
    while not done:
        action = agent.sample_action(obs)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        agent.rewards.append(reward)
        done = terminated or truncated

    reward_over_episodes.append(wrapped_env.return_queue[-1])# 最后一个元素是当前episode的奖励
    agent.update()

    if episode % 500 == 0:
        avg_reward = int(np.mean(wrapped_env.return_queue))
        print("Episode:", episode, "Average Reward:", avg_reward)

# rewards_over_seeds.append(reward_over_episodes)

# df1 = pd.DataFrame(rewards_over_seeds).melt()
# df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
# sns.set(style="darkgrid", context="talk", palette="rainbow")
# sns.lineplot(x="episodes", y="reward", data=df1).set(
#     title="REINFORCE for InvertedPendulum-v4"
# )
# plt.show()

sns.lineplot(x=range(len(reward_over_episodes)),
             y=reward_over_episodes)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("REINFORCE Training Curve")

plt.show()

save_checkpoint(agent, reward_over_episodes )

wrapped_env.close()
def run(ai):
    env1 = gym.make("InvertedPendulum-v4",render_mode="human")
    wrapped_env1 = gym.wrappers.RecordEpisodeStatistics(env1, 50)  # Records episode-reward
    
    obs, info = wrapped_env1.reset(seed=8)
    
    done = False
    while not done:
        with torch.no_grad():  # 禁用梯度
            action = ai.sample_action(obs)
        obs, reward, terminated, truncated, info = wrapped_env1.step(action)
        ai.rewards.append(reward)
        done = terminated or truncated

    final_reward = wrapped_env.return_queue[-1]
    print("Final Reward:", final_reward)
    os.system("pause")
    wrapped_env1.close()

#reward_over_episodes, start_episode, seed = load_checkpoint(agent, path="checkpoint.pth")

run(agent)
