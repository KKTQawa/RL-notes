import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from collections import deque
import os

# 设置随机种子以确保实验可重复性
seed = 42  # 随机种子，用于控制随机数生成，确保实验结果可复现
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，确保所有 GPU 的随机性一致

# 设置计算设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 检查是否有 GPU 可用
device = 'cpu'  # 强制使用 CPU，可根据需要改为 GPU，注意 GPU 不一定总是更快（数据传输开销）

# 策略网络（Actor），用于生成确定性动作
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 第二个隐藏层到输出层
        self.action_bound = action_bound  # 动作范围的最大值，例如 Pendulum 的 [-2, 2]

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数增加非线性
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound  # 输出通过 tanh 缩放到 [-1, 1]，再乘以 action_bound

# Q 值网络（Critic），用于估计状态-动作对的价值
class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)  # 输入层拼接状态和动作
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc_out = nn.Linear(hidden_dim, 1)  # 输出单个 Q 值

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=-1)  # 在最后一个维度拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # 输出 Q 值

# 经验回放缓冲区，用于存储和采样经验
class ReplayMemory:
    def __init__(self, maxlen, seed=42):
        self.memory = deque(maxlen=maxlen)  # 双端队列，容量满时自动移除旧数据
        if seed is not None:
            random.seed(seed)  # 设置随机种子，确保采样一致性

    def append(self, transition):
        self.memory.append(transition)  # 添加新的经验元组 (s, a, s', r, done)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)  # 随机采样指定数量的经验

    def __len__(self):
        return len(self.memory)  # 返回当前缓冲区大小

def train():
    # === 超参数设置 ===
    learning_rate_actor = 0.0003  # Actor 学习率，控制策略更新速度
    learning_rate_critic = 0.001  # Critic 学习率，控制 Q 值更新速度
    discount_factor_g = 0.99  # 折扣因子 γ，用于计算未来奖励的折现，公式：Q(s,a) = r + γ * Q(s',a')
    hidden_dim = 128  # 隐藏层神经元数量，决定网络容量
    max_steps_per_episode = 10000  # 每个回合的最大步数上限
    buffer_size = 100000  # 经验回放缓冲区容量
    batch_size = 128  # 每次训练的批量大小
    replay_buffer = ReplayMemory(buffer_size)  # 初始化经验回放缓冲区
    tau = 0.005  # 软更新参数，用于目标网络更新，公式：θ_target = τ*θ + (1-τ)*θ_target
    noise_std = 0.1  # 探索时添加的高斯噪声标准差
    policy_noise = 0.2  # 目标策略平滑噪声标准差，用于正则化
    noise_clip = 0.5  # 目标策略噪声裁剪范围，限制噪声大小
    policy_freq = 2  # 延迟策略更新的频率，Critic 更新 2 次后更新 1 次 Actor

    # === 初始化记录变量 ===
    best_reward = -float('inf')  # 跟踪历史最佳奖励
    rewards_all_episodes = []  # 存储每个回合的总奖励
    episodes_iter = 0  # 回合计数器
    update_iter = 0  # 更新计数器，用于控制延迟更新

    # === 创建环境 ===
    env = gym.make('Pendulum-v1')  # 创建 Pendulum-v1 环境，一个单摆控制任务
    states_dim = env.observation_space.shape[0]  # 状态空间维度，例如 [cosθ, sinθ, θ_dot]
    actions_dim = env.action_space.shape[0]  # 动作空间维度，例如 [torque]
    action_bound = env.action_space.high[0]  # 动作范围的最大值，例如 2.0
    print(f'状态空间维度: {states_dim}, 动作空间维度: {actions_dim}')

    # === 初始化网络 ===
    actor = PolicyNet(states_dim, actions_dim, hidden_dim, action_bound).to(device)  # 主策略网络
    actor_target = PolicyNet(states_dim, actions_dim, hidden_dim, action_bound).to(device)  # 目标策略网络
    actor_target.load_state_dict(actor.state_dict())  # 初始化时同步参数

    critic_1 = QValueNet(states_dim, actions_dim, hidden_dim).to(device)  # 主 Critic 网络 1
    critic_1_target = QValueNet(states_dim, actions_dim, hidden_dim).to(device)  # 目标 Critic 网络 1
    critic_1_target.load_state_dict(critic_1.state_dict())
    
    critic_2 = QValueNet(states_dim, actions_dim, hidden_dim).to(device)  # 主 Critic 网络 2
    critic_2_target = QValueNet(states_dim, actions_dim, hidden_dim).to(device)  # 目标 Critic 网络 2
    critic_2_target.load_state_dict(critic_2.state_dict())

    # === 定义优化器和损失函数 ===
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_actor)  # Actor 的 Adam 优化器
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=learning_rate_critic)  # Critic 1 的优化器
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=learning_rate_critic)  # Critic 2 的优化器
    critic_loss_fn = nn.MSELoss()  # 均方误差损失，用于 Critic 更新，公式：L = (Q - Q_target)^2

    # === 训练主循环 ===
    for episode in range(200):  # 无限循环，直到手动停止
        state = env.reset()[0]  # 重置环境，获取初始状态
        state = torch.tensor(state, dtype=torch.float32, device=device)  # 转换为张量
        rewards_one_episode = 0  # 当前回合的总奖励
        terminated = False  # 是否终止
        truncated = False  # 是否截断

        # === 收集一个回合的经验 ===
        while not (terminated or truncated) and rewards_one_episode < 100:
            with torch.no_grad():  # 采样时不计算梯度，节省资源
                action = actor(state).flatten()  # Actor 输出动作
                noise = np.random.normal(0, noise_std, size=actions_dim)  # 添加高斯噪声以探索
                action = np.clip(action.cpu().numpy() + noise, -action_bound, action_bound)  # 裁剪动作
                action = torch.tensor(action, dtype=torch.float32, device=device)  # 转换回张量

            next_state, reward, terminated, truncated, _ = env.step(action.numpy())  # 执行动作
            rewards_one_episode += reward  # 累加奖励

            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)  # 下一状态
            reward = torch.tensor([reward], dtype=torch.float32, device=device)  # 奖励
            terminated = torch.tensor([terminated], dtype=torch.float32, device=device)  # 终止标志

            replay_buffer.append((state, action, next_state, reward, terminated))  # 存入经验缓冲区
            state = next_state  # 更新当前状态

            # === 开始训练 ===
            if len(replay_buffer) > batch_size:  # 缓冲区足够大时开始训练
                update_iter += 1  # 更新计数器递增
                mini_batch = replay_buffer.sample(batch_size)  # 随机采样
                states, actions, new_states, rewards, terminations = zip(*mini_batch)  # 解包

                states = torch.stack(states)  # 转换为张量堆栈
                actions = torch.stack(actions)
                new_states = torch.stack(new_states)
                rewards = torch.stack(rewards)
                terminations = torch.stack(terminations)

                # === TD3 核心要素 1：目标策略平滑正则化 ===
                # 数学公式：a' = π(s') + ε, ε ~ clip(N(0, σ), -c, c)
                # 说明：在目标动作上添加噪声，模拟环境不确定性，防止策略过拟合
                with torch.no_grad():
                    noise = torch.normal(0, policy_noise, size=actions.shape).to(device)  # 生成噪声
                    noise = torch.clamp(noise, -noise_clip, noise_clip)  # 裁剪噪声
                    next_actions = actor_target(new_states)  # 目标策略生成动作
                    next_actions = torch.clamp(next_actions + noise, -action_bound, action_bound)  # 添加噪声并裁剪

                    # === TD3 核心要素 2：双重 Q 值计算 ===
                    # 数学公式：Q_target = r + γ * min(Q1'(s', a'), Q2'(s', a'))
                    # 说明：使用两个 Critic 网络，取最小值减少 Q 值过估计
                    target_q1 = critic_1_target(new_states, next_actions)  # Critic 1 的目标 Q 值
                    target_q2 = critic_2_target(new_states, next_actions)  # Critic 2 的目标 Q 值
                    target_q = torch.min(target_q1, target_q2)  # 取最小值
                    q_targets = rewards + discount_factor_g * target_q * (1 - terminations)  # Bellman 方程

                # === 更新 Critic 网络 ===
                q1_values = critic_1(states, actions)  # 当前 Q 值估计
                q2_values = critic_2(states, actions)
                critic_1_loss = critic_loss_fn(q1_values, q_targets)  # Critic 1 损失
                critic_2_loss = critic_loss_fn(q2_values, q_targets)  # Critic 2 损失

                critic_1_optimizer.zero_grad()  # 清零梯度
                critic_1_loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(critic_1.parameters(), 1.0)  # 梯度裁剪，防止爆炸
                critic_1_optimizer.step()  # 更新参数

                critic_2_optimizer.zero_grad()
                critic_2_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_2.parameters(), 1.0)
                critic_2_optimizer.step()

                # === TD3 核心要素 3：延迟策略更新 ===
                # 说明：Critic 更新多次后才更新 Actor，增强稳定性
                if update_iter % policy_freq == 0:
                    # 数学公式：L_actor = -mean(Q1(s, π(s)))
                    # 说明：最大化 Q1 值来更新 Actor，仅使用 Critic 1
                    actor_loss = -critic_1(states, actor(states)).mean()  # Actor 损失
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_optimizer.step()

                    # === 软更新目标网络 ===
                    # 数学公式：θ_target = τ*θ + (1-τ)*θ_target
                    for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for target_param, param in zip(critic_1_target.parameters(), critic_1.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    for target_param, param in zip(critic_2_target.parameters(), critic_2.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        rewards_all_episodes.append(rewards_one_episode)  # 记录回合奖励
        episodes_iter += 1

        # === 记录和可视化 ===
        if episodes_iter % 20 == 0:
            if rewards_one_episode > best_reward:
                best_reward = rewards_one_episode
            mean_reward = np.mean(rewards_all_episodes[-20:])  # 最近 20 回合平均奖励
            print(f'回合: {episodes_iter}, 最佳奖励: {best_reward}, 平均奖励: {mean_reward:.3f}')

            mean_rewards = [np.mean(rewards_all_episodes[max(0, t - 20):t + 1]) for t in range(episodes_iter)]
            plt.plot(mean_rewards)  # 绘制平均奖励曲线
            plt.savefig('TD3-Pendulum_train.png')
            plt.close()

            # 保存模型
            torch.save(actor.state_dict(), "TD3-Pendulum_actor_net.pt")
            # torch.save(critic_1.state_dict(), "TD3-Pendulum_critic_1_net.pt")
            # torch.save(critic_2.state_dict(), "TD3-Pendulum_critic_2_net.pt")
#Pendulum 坚持倒立越久，奖励越高 默认最大步数为 200
def test():
    # === 测试函数 ===
    env = gym.make('Pendulum-v1', render_mode='human')  # 创建测试环境，可视化
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    hidden_dim = 128

    actor = PolicyNet(state_dim, action_dim, hidden_dim, action_bound).to(device)
    try:
        actor.load_state_dict(torch.load('TD3-Pendulum_actor_net.pt', map_location=device))
        print("成功加载训练好的 Actor 模型")
    except FileNotFoundError:
        print("模型文件 'TD3-Pendulum_actor_net.pt' 未找到，请先运行 train() 进行训练")
        return

    actor.eval()  # 设置为评估模式

    for i in range(5):  # 测试 5 个回合
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            env.render()  # 显示环境
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action = actor(state_tensor).cpu().numpy()  # 生成动作
                action = np.clip(action, -action_bound, action_bound)  # 裁剪动作

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated

        print(f"测试回合 {i+1}, 总奖励: {total_reward:.1f}")
    os.system('pause')
    env.close()

if __name__ == '__main__':
    #train(isLoad=False)  # 默认运行训练
    test()  # 注释掉测试，训练后可手动取消注释运行