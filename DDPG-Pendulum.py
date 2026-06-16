import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from sympy import false
from torch import nn
import torch.nn.functional as F
import itertools
from torch.distributions import Normal
from collections import deque
import os

# 设置随机种子以确保实验可重复性
seed = 42  # 随机种子，可选择任意数字
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU 训练

# 设置计算设备（优先使用 GPU，若无则使用 CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # 强制使用 CPU，有时由于数据传输开销，GPU 并非总是更快

# 策略网络（Actor），输出确定性动作
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 动作范围的最大值，例如 Pendulum 的 [-2, 2]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound  # 输出范围 [-action_bound, action_bound]

# Q 值网络（Critic），估计状态-动作对的价值
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=-1)  # 在最后一个维度拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # 输出 Q 值

# 经验回放缓冲区
class ReplayMemory:
    def __init__(self, maxlen, seed=42):
        self.memory = deque(maxlen=maxlen)  # 双端队列，超出容量时自动移除旧数据
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)  # 随机采样

    def __len__(self):
        return len(self.memory)

def train():
    # 超参数设置
    learning_rate_actor = 0.0003  # Actor 网络学习率，控制策略更新速度
    learning_rate_critic = 0.001  # Critic 网络学习率，控制价值估计更新速度
    gamma = 0.99  # 折扣因子 gamma，用于计算未来奖励的折现

    hidden_dim = 128  # 隐藏层神经元数量，决定网络容量
    max_steps_per_episode = 10000  # 每个回合的最大步数上限

    buffer_size = 100000
    batch_size = 128
    replay_buffer = ReplayMemory(buffer_size)

    tau = 0.005 #软更新参数
    noise_std = 0.1  # 高斯噪声的标准差，用于探索

    # 初始化记录变量
    best_reward = -float('inf')  # 跟踪历史最佳奖励
    rewards_all_episodes = []  # 存储每个回合的总奖励
    episodes_iter = 0  # 回合计数器

    env = gym.make('Pendulum-v1')
    states_dim = env.observation_space.shape[0]
    actions_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    print(f'状态空间维度: {states_dim}, 动作空间维度: {actions_dim}')

    # 初始化 Actor 和 Critic 网络并移到指定设备
    actor = ActorNet(states_dim, actions_dim, hidden_dim, action_bound).to(device)
    actor_target = ActorNet(states_dim, actions_dim, hidden_dim, action_bound).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = CriticNet(states_dim, actions_dim, hidden_dim).to(device)
    critic_target = CriticNet(states_dim, actions_dim, hidden_dim).to(device) 
    critic_target.load_state_dict(critic.state_dict())
    # 定义优化器和损失函数
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate_critic)

    # 训练主循环
    for episode in range(140):
        state = env.reset()[0]  
        state = torch.tensor(state, dtype=torch.float32, device=device)
        rewards_one_episode = 0  # 当前回合的总奖励

        terminated = False
        truncated = False

        # 收集一个回合的轨迹数据
        for t in range(max_steps_per_episode):
        #while not( terminated or truncated) and rewards_one_episode < 100:
            with torch.no_grad():  # 采样时不计算梯度，节省计算资源
                action = actor(state).flatten()#flatten():展平输出，将动作转换为一维张量
                noise = np.random.normal(0, noise_std, size=actions_dim)  # 高斯噪声
                action = np.clip(action.cpu().numpy() + noise, -action_bound, action_bound)#将动作限制在 [-action_bound, action_bound] 范围内
                action = torch.tensor(action, dtype=torch.float32, device=device)  # 转换回张量

            next_state, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            rewards_one_episode += reward

            # 将数据转换为张量并存储到轨迹中
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).detach()
            reward = torch.tensor([reward], dtype=torch.float32, device=device).detach()
            terminated = torch.tensor([terminated], dtype=torch.float32, device=device).detach()
            truncated = torch.tensor([truncated], dtype=torch.float32, device=device).detach()

            replay_buffer.append((state, action.detach(), next_state, reward, terminated, truncated))

            state = next_state  # 更新状态

            # 如果缓冲区足够大，开始训练
            if len(replay_buffer) > batch_size:
                mini_batch = replay_buffer.sample(batch_size)
                states, actions, new_states, rewards, terminations, truncated = zip(*mini_batch)

                states = torch.stack(states)  #直接变成  shape = (mini_batch_size, states.n=3)
                actions = torch.stack(actions)
                new_states = torch.stack(new_states)
                rewards = torch.stack(rewards)
                terminations = torch.stack(terminations)
                truncated = torch.stack(truncated)

                # 计算目标 Q 值
                with torch.no_grad():
                    next_actions = actor_target(new_states)
                    next_Q = critic_target(new_states, next_actions)
                    target_Q = rewards + gamma * next_Q * (1 - done)  # Bellman 方程
                
                # 更新 Critic
                q_values = critic(states, actions)
                critic_loss = nn.MSELoss()(q_values, target_Q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)  # 梯度裁剪 不能超过1.0*max梯度
                critic_optimizer.step()

                # 更新 Actor
                actor_loss = -critic(states, actor(states)).mean()  # 最大化 Q 值
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                # 软更新目标网络
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break
        rewards_all_episodes.append(rewards_one_episode)
        episodes_iter += 1

        # 记录和可视化训练结果
        if episodes_iter % 20 == 0:
            if rewards_one_episode > best_reward:
                best_reward = rewards_one_episode
            mean_reward = np.mean(rewards_all_episodes[-20:])  # 计算最近 20 回合平均奖励
            print(f'回合: {episodes_iter}, 最佳奖励: {best_reward}, 平均奖励: {mean_reward:.3f}')

            # 绘制平均奖励曲线
            mean_rewards = [np.mean(rewards_all_episodes[max(0, t - 20):t + 1]) for t in range(episodes_iter)]
            plt.plot(mean_rewards)
            plt.savefig('DDPG-Pendulum_train.png')
            plt.close()

            # 保存模型参数
            torch.save(actor.state_dict(), "DDPG-Pendulum_actor_net.pt")
            #torch.save(critic.state_dict(), "DDPG-Pendulum_critic_net.pt")

def test():
    # 创建测试环境
    env = gym.make('Pendulum-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    hidden_dim = 128

    # 初始化 Actor 网络并加载模型
    actor = ActorNet(state_dim, action_dim, hidden_dim, action_bound).to(device)
    try:
        actor.load_state_dict(torch.load('DDPG-Pendulum_actor_net.pt', map_location=device))
        print("成功加载训练好的 Actor 模型")
    except FileNotFoundError:
        print("模型文件 'DDPG-Pendulum_actor_net.pt' 未找到，请先运行 train() 进行训练")
        return

    actor.eval()  # 设置为评估模式

    # 运行 5 个测试回合
    for i in range(5):
        state = env.reset()[0]#初始位置是随机的
        print("环境已重置")
        os.system("pause")
        total_reward = 0
        done = False

        while not done:
            #env.render()  # 显示环境
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action = actor(state_tensor).cpu().numpy()  # DDPG 输出确定性动作
                action = np.clip(action, -action_bound, action_bound)

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated

        print(f"测试回合 {i+1}, 总奖励: {total_reward:.1f}")
    os.system("pause")

    env.close()

if __name__ == '__main__':
    train()
    test()