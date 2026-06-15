import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from torch.distributions import Normal
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

# 连续动作策略网络（Actor），输出动作的均值和标准差
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层
        #高斯分布：均值、方差
        self.fc_mean = nn.Linear(hidden_dim, action_dim)  # 输出动作均值
        # 使用可训练参数定义对数标准差，确保标准差为正
        self.fc_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        """
        前向传播：
        - 输入状态 x，经过隐藏层 fc1 和 ReLU 激活函数
        - 输出动作分布的均值 μ 和标准差 σ
        - μ = tanh(fc_mean(x))，范围 [-1, 1]，需进一步缩放到动作空间
        - σ = exp(fc_log_std)，确保标准差为正数
        """
        x = F.relu(self.fc1(x))
        mean = torch.tanh(self.fc_mean(x))  # 均值 μ，范围 [-1, 1]
        log_std = self.fc_log_std.expand_as(mean)  # 对数标准差扩展到与 mean 相同形状
        std = torch.exp(log_std.clamp(-20, 2))  # σ = exp(log_std)，限制范围避免数值爆炸
        return mean, std


# 定义价值网络类（Critic），用于估计状态价值 V(s)
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出层，输出单一状态值

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return self.fc2(x)  # 输出状态价值 V(s)

def train():
    # 超参数设置
    learning_rate_actor = 0.0003  # Actor 网络学习率，控制策略更新速度
    learning_rate_critic = 0.001  # Critic 网络学习率，控制价值估计更新速度
    gamma = 0.99  # 折扣因子 gamma，用于计算未来奖励的折现

    clip_epsilon = 0.2  # PPO 剪切参数，限制新旧策略的更新幅度
    gae_lambda = 0.95  # GAE 的 λ 参数，平衡偏差与方差
    ppo_epochs = 10  # 每个回合内对采集的数据进行多次优化

    max_steps_per_episode = 10000  # 每个回合的最大步数上限
    hidden_dim = 128  # 隐藏层神经元数量，决定网络容量
    action_scale = 2.0  # Pendulum 动作范围 [-2, 2] 的缩放因子

    # 初始化记录变量
    best_reward = -float('inf')  # 跟踪历史最佳奖励
    rewards_all_episodes = []  # 存储每个回合的总奖励
    episodes_iter = 0  # 回合计数器

    # 创建 Pendulum-v1 环境
    env = gym.make('Pendulum-v1')
    states_dim = env.observation_space.shape[0]  # 状态空间维度（如位置、速度等）
    actions_dim = env.action_space.shape[0]  # 动作空间维度（如左、右）
    print(f'状态空间维度: {states_dim}, 动作空间维度: {actions_dim}')

    # 初始化 Actor 和 Critic 网络并移到指定设备
    actor_net = ActorNet(states_dim, actions_dim, hidden_dim).to(device)
    critic_net = ValueNet(states_dim, hidden_dim).to(device)

    # 定义优化器和损失函数
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=learning_rate_actor)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=learning_rate_critic)
    critic_loss_fn = nn.MSELoss()  # Critic 使用均方误差损失，优化价值估计

    # 训练主循环
    for episode in range(800):
        state = env.reset()[0]  # 重置环境，获取初始状态
        state = torch.tensor(state, dtype=torch.float, device=device)
        rewards_one_episode = 0  # 当前回合的总奖励
        transition_dict = {'states': [], 'actions': [], 'log_probs': [], 'next_states': [], 'rewards': [], 'terminated': [],'truncated':[]}

        # 收集一个回合的轨迹数据
        #按轮次更新
        with torch.no_grad():  # 采样时不计算梯度，节省计算资源
            for t in range(max_steps_per_episode):
                mean, std = actor_net(state)
                dist = Normal(mean, std)
                u = dist.sample()
                # # 计算对数概率：log π(a|s) = log N(a|μ, σ)-log(1-tanh^2(N(a|μ, σ)))
                #PPO/SAC动作修正
                # log_prob = dist.log_prob(u).sum(dim=-1)
                log_prob = dist.log_prob(u) - torch.log(1 - torch.tanh(u).pow(2) + 1e-6)
                log_prob = log_prob.sum(-1)
                # # 将动作缩放到 [-2, 2]
                a = torch.tanh(u) * action_scale

                # print(f'action {action}')
                # print(f'action_clipped {action_clipped}')
                # print(f'action_clipped.numpy() {action_clipped.numpy()}')
                # exit()
                next_state, reward, terminated, truncated, _ = env.step(a.numpy())

                rewards_one_episode += reward

                # 将数据转换为张量并存储到轨迹中
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                terminated = torch.tensor(terminated, dtype=torch.float, device=device)
                truncated = torch.tensor(truncated, dtype=torch.float, device=device)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(a)
                transition_dict['rewards'].append(reward)
                transition_dict['next_states'].append(next_state)
                transition_dict['terminated'].append(terminated)
                transition_dict['truncated'].append(truncated)
                transition_dict['log_probs'].append(log_prob)


                state = next_state  # 更新状态
                if terminated or truncated:  # 回合结束条件
                    break

        rewards_all_episodes.append(rewards_one_episode)
        episodes_iter += 1

        # 将轨迹数据转换为张量堆栈，便于批量处理
        states = torch.stack(transition_dict['states'])
        actions = torch.stack(transition_dict['actions'])
        rewards = torch.stack(transition_dict['rewards'])
        next_states = torch.stack(transition_dict['next_states'])
        terminated = torch.stack(transition_dict['terminated'])
        truncated = torch.stack(transition_dict['truncated'])
        old_log_probs = torch.stack(transition_dict['log_probs'])
        done = (terminated.bool() | truncated.bool()).float()

        # 计算 GAE 优势函数和回报
        values = critic_net(states).squeeze()# 当前状态的价值估计
        next_values = critic_net(next_states).squeeze() # 下一状态的价值估计
        # 时序差分误差 (TD error): δ = r + γ * V(s') * (1 - done) - V(s)
        deltas = rewards + gamma * next_values * (1 - done) - values
        advantages = []
        advantage = 0
        # GAE 计算：从后向前累积优势
        for delta in reversed(deltas.tolist()):
            # 应用GAE核心公式：a_t = δ_t + γ * λ * a_{t+1}
            advantage = delta + gamma * gae_lambda * advantage
            advantages.insert(0, advantage)
        advantages = torch.tensor(advantages, dtype=torch.float, device=device)

        # 标准化优势函数，减小数值范围，提高训练稳定性：A = (A - mean(A)) / (std(A) + ε)
        advantages = (advantages - advantages.mean()) /  (advantages.std() + 1e-8)

        # PPO 更新
        for _ in range(ppo_epochs):  # 对同一批数据进行多次优化
            # 计算新策略的概率分布
            mean, std = actor_net(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            # print(f'states {states.shape}')
            # print(f'actions {actions.shape}')
            # print(f'new_log_probs {new_log_probs.shape}')
            # exit()


            """
            PPO 核心公式：
            - r(θ) = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ(a|s) - log π_θ_old(a|s))  # 概率比
            - L_clip = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]  # 剪切代理目标
            - L_vf = E[(V(s) - R)^2]  # Critic 的均方误差损失
            """
            #计算actor 损失
            ratios = torch.exp(new_log_probs - old_log_probs)  # r(θ)
            surr1 = ratios * advantages # 未剪切的代理目标
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages  # 剪切后的代理目标
            actor_loss = -torch.mean( torch.min(surr1, surr2) )  #返回的是一个数（标量）,取最小值，确保更新幅度受限

            # Critic 损失：优化价值函数逼近回报
            returns = advantages + values
            critic_loss = critic_loss_fn( critic_net(states).squeeze(), returns.detach() )

            # 反向传播和参数更新
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        # 记录和可视化训练结果
        if episodes_iter % 20 == 0:
            if rewards_one_episode > best_reward:
                best_reward = rewards_one_episode
            mean_reward = np.mean(rewards_all_episodes[-20:])  # 计算最近 20 回合平均奖励
            print(f'回合: {episodes_iter}, 最佳奖励: {best_reward}, 平均奖励: {mean_reward:.3f}')

            # 绘制平均奖励曲线
            mean_rewards = [np.mean(rewards_all_episodes[max(0, t - 20):t + 1]) for t in range(episodes_iter)]
            plt.plot(mean_rewards)
            plt.savefig('PPO-Pendulum_train.png')
            plt.close()

            # 保存模型参数
            torch.save(actor_net.state_dict(), "PPO-Pendulum_actor_net.pt")
            #torch.save(critic_net.state_dict(), "PPO-Pendulum_critic_net.pt")

# 测试函数：使用训练好的模型评估性能
# Pendulum-v1 坚持倒立越久，奖励越高 默认步数：200
def test():
    # 测试函数
    env = gym.make('Pendulum-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_net = ActorNet(state_dim, action_dim, 128).to(device)
    try:
        actor_net.load_state_dict(torch.load('PPO-Pendulum_actor_net.pt', map_location=device))
    except FileNotFoundError:
        print("模型文件 PPO-Pendulum_actor_net 未找到，请先运行 train() 进行训练")
        return

    actor_net.eval()

    for _ in range(5):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                mean, std = actor_net(state_tensor)
                action = torch.tanh(mean) * 2.0  # 使用均值作为确定性动作
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done=terminated or truncated
            total_reward += reward
            state = next_state

        print(f"测试回合奖励: {total_reward:.1f}")

    os.system('pause')


    env.close()

if __name__ == '__main__':
    train()
    test()