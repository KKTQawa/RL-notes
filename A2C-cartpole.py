import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import os
# 设置随机种子以确保实验可重复性
seed = 42  # 随机种子，可选择任意数字
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU 训练

# 设置计算设备（优先使用 GPU，若无则使用 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # 强制使用 CPU，有时由于数据传输开销，GPU 并非总是更快

# 定义策略网络类（Actor），用于输出动作概率分布
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层的全连接层
        self.fc2 = nn.Linear(hidden_dim, action_dim)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数增加非线性
        return F.softmax(self.fc2(x), dim=-1)  # 使用 Softmax 输出动作概率分布

# 定义价值网络类（Critic），用于估计状态价值 V(s)
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 第一层全连接层
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出层，输出状态值 V(s)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return self.fc2(x)  # 输出状态价值

# A2C 训练函数，无终止条件，需手动停止（通过查看奖励曲线判断）
def train(enable_double_dqn=True, render=False):
    # 超参数设置
    learning_rate_actor = 0.001  # Actor 网络学习率，控制策略更新速度
    learning_rate_critic = 0.01  # Critic 网络学习率，控制价值估计更新速度
    discount_factor_g = 0.98  # 折扣因子 gamma，用于计算未来奖励的折现
    target_update_freq = 10  # 目标网络更新频率（仅在启用 Double DQN 时有效）
    hidden_dim = 128  # 隐藏层神经元数量，决定网络容量
    max_steps_per_episode = 1000  # 每个回合的最大步数上限

    # 初始化记录变量
    best_reward = -float("inf")  # 跟踪历史最佳奖励
    rewards_all_episodes = []  # 存储每个回合的总奖励
    episodes_iter = 0  # 回合计数器

    # 创建 CartPole-v1 环境
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    num_states = env.observation_space.shape[0]  # 状态空间维度（如位置、速度等）
    num_actions = env.action_space.n  # 动作空间维度（如左、右）
    print(f"状态空间维度: {num_states}, 动作空间维度: {num_actions}")

    # 初始化 Actor 和 Critic 网络并移到指定设备
    actor_net = PolicyNet(num_states, num_actions, hidden_dim).to(device)
    critic_net = ValueNet(num_states, hidden_dim).to(device)
    target_critic_net = ValueNet(num_states, hidden_dim).to(device)  # 目标 Critic 网络
    target_critic_net.load_state_dict(critic_net.state_dict())  # 初始化目标网络参数
    target_critic_net.eval()  # 设置为评估模式，不计算梯度

    # 定义优化器和损失函数
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=learning_rate_actor)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=learning_rate_critic)
    critic_loss_fn = nn.MSELoss()  # Critic 使用均方误差损失，优化价值估计

    # 训练主循环（无限回合，直到手动停止）
    for episode in range(500):
        state = env.reset()[0]  # 重置环境，获取初始状态
        state = torch.tensor(state, dtype=torch.float, device=device)
        rewards_one_episode = 0  # 当前回合的总奖励
        transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "terminated": []}

        # 收集一个回合的轨迹数据
        with torch.no_grad():  # 采样时不计算梯度，节省计算资源
            for t in range(max_steps_per_episode):
                probs = actor_net(state)  # 获取当前状态下的动作概率分布
                action_dist = torch.distributions.Categorical(probs)  # 定义分类分布
                action = action_dist.sample()  # 从概率分布中采样动作

                # 执行动作，获取环境反馈
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                rewards_one_episode += reward

                # 将数据转换为张量并存储到轨迹中
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                terminated = torch.tensor(terminated, dtype=torch.float, device=device)

                transition_dict["states"].append(state)
                transition_dict["actions"].append(action)
                transition_dict["rewards"].append(reward)
                transition_dict["next_states"].append(next_state)
                transition_dict["terminated"].append(terminated)

                state = next_state  # 更新状态
                if terminated or truncated:  # 回合结束条件
                    break

        rewards_all_episodes.append(rewards_one_episode)
        episodes_iter += 1

        # 将轨迹数据转换为张量堆栈，便于批量处理
        states = torch.stack(transition_dict["states"])
        actions = torch.stack(transition_dict["actions"])
        rewards = torch.stack(transition_dict["rewards"])
        next_states = torch.stack(transition_dict["next_states"])
        terminated = torch.stack(transition_dict["terminated"])

        # 计算时序差分目标（TD Target）和优势估计
        with torch.no_grad():
            # 使用目标网络计算下一状态的价值 V(s『)
            next_values = target_critic_net(next_states).squeeze()
            # TD 目标：r + γ * V(s』) * (1 - done)
            td_target = rewards + discount_factor_g * next_values * (1 - terminated)
        # 当前状态的价值估计 V(s)
        values = critic_net(states).squeeze()
        # 优势函数 A(s, a) = TD_target - V(s)
        advantages = td_target - values

        # 计算 Actor 损失（策略梯度）
        # log π(a|s)：对策略网络输出的动作概率取对数
        log_probs = torch.log(actor_net(states).gather(1, actions.unsqueeze(1))).squeeze()
        # Actor 损失：-log π(a|s) * A(s, a)，负号表示最大化目标
        actor_loss = torch.mean(-log_probs * advantages.detach())

        # 计算 Critic 损失
        # Critic 的目标是使 V(s) 逼近 TD 目标
        critic_loss = critic_loss_fn(values, td_target.detach())

        # 更新网络参数
        actor_optimizer.zero_grad()  # 清空 Actor 梯度
        critic_optimizer.zero_grad()  # 清空 Critic 梯度
        actor_loss.backward()  # 计算 Actor 损失的梯度
        critic_loss.backward()  # 计算 Critic 损失的梯度
        actor_optimizer.step()  # 更新 Actor 参数
        critic_optimizer.step()  # 更新 Critic 参数

        # 更新目标网络（软更新或硬更新）
        if episodes_iter % target_update_freq == 0:
            target_critic_net.load_state_dict(critic_net.state_dict())  # 硬更新目标网络

        # 记录和可视化训练结果
        if episodes_iter % 20 == 0:
            if rewards_one_episode > best_reward:
                best_reward = rewards_one_episode
            mean_reward = np.mean(rewards_all_episodes[-20:])  # 计算最近 20 回合平均奖励
            print(f"回合: {episodes_iter}, 最佳奖励: {best_reward}, 平均奖励: {mean_reward:.3f}")

            # 绘制平均奖励曲线
            mean_rewards = [np.mean(rewards_all_episodes[max(0, t - 20):t + 1]) for t in range(episodes_iter)]
            plt.plot(mean_rewards)
            plt.savefig("A2C-cartpole_train.png")
            plt.close()

            # 保存模型参数
            torch.save(actor_net.state_dict(), "A2C-cartpole_actor_net.pt")
            #torch.save(critic_net.state_dict(), "A2C-cartpole_critic_net.pt")

# 测试函数：使用训练好的模型评估性能
def test(render=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    policy_net = PolicyNet(num_states, num_actions, 128)
    try:
        policy_net.load_state_dict(torch.load("A2C-cartpole_actor_net.pt"))  # 加载训练好的模型
    except FileNotFoundError:
        print("模型文件未找到，请先进行训练。")
        return

    policy_net.eval()  # 设置为评估模式（禁用 dropout 等）
    total_rewards = 0
    episodes = 5

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        done = False
        rewards_one_episode = 0

        while not done:
            with torch.no_grad():
                probs = policy_net(state)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()  # 采样动作

            new_state, reward, terminated, truncated, _ = env.step(action)
            rewards_one_episode += reward
            state = torch.tensor(new_state, dtype=torch.float).unsqueeze(0)
            done = terminated or truncated

        total_rewards += rewards_one_episode
        print(f"回合 {episode + 1}: 奖励 = {rewards_one_episode}")

    avg_reward = total_rewards / episodes
    print(f"{episodes} 次测试的平均奖励: {avg_reward:.3f}")
    os.system("pause")

if __name__ == "__main__":
    train()
    test()