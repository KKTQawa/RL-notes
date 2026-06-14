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
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # 强制使用CPU，避免硬件差异影响结果


# === 策略网络（Actor） ===
# SAC中策略网络输出动作分布的均值和标准差，用于探索环境
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        """
        初始化策略网络
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度
        :param hidden_dim: 隐藏层维度
        :param action_bound: 动作范围边界（如Pendulum中为[-2, 2]）
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)  # 输出动作均值 μ
        self.log_std = nn.Linear(hidden_dim, action_dim)  # 输出动作标准差的对数 log σ
        self.action_bound = action_bound

        # 参数初始化
        nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)  # 均值权重初始化为小范围均匀分布
        nn.init.constant_(self.log_std.bias, -1.0)  # 标准差偏置初始化为-1，使初始标准差较小

    def forward(self, x):
        """
        前向传播，计算动作分布的均值和标准差
        :param x: 输入状态张量
        :return: 均值 mu 和对数标准差 log_std
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x)) * self.action_bound  # 均值通过tanh缩放到动作范围
        log_std = torch.clamp(self.log_std(x), -20, 2)  # 限制log_std范围，避免数值不稳定
        return mu, log_std

    def sample(self, state):
        """
        从策略分布中采样动作，并计算对应的对数概率
        数学公式：
        a ~ N(μ, σ^2), 其中 σ = exp(log_std)
        log π(a|s) = log N(a|μ, σ^2) - log(1 - tanh^2(a))（调整项）
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()  # 标准差 σ = e^(log_std)

        # 重参数化技巧：a = μ + σ * ε, 其中 ε ~ N(0, 1)
        noise = torch.randn_like(mu)
        action = mu + noise * std

        # 计算对数概率（高斯分布的对数概率公式）
        log_prob = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
        # 修正项：由于tanh压缩，需加入边界调整
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(-1, keepdim=True)
        return torch.tanh(action) * self.action_bound, log_prob  # 返回最终动作和对数概率


# === 双Q值网络（Critic） ===
# SAC使用双Q网络减少过估计偏差
class TwinQValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        初始化双Q值网络
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        # Q1网络
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2网络
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        前向传播，计算两个Q值
        :param state: 状态张量
        :param action: 动作张量
        :return: Q1(s,a) 和 Q2(s,a)
        """
        sa = torch.cat([state, action], -1)  # 拼接状态和动作
        return self.q1(sa), self.q2(sa)


# === 经验回放缓冲区 ===
class ReplayMemory:
    def __init__(self, maxlen, seed=42):
        """
        初始化经验回放缓冲区
        :param maxlen: 缓冲区最大容量
        :param seed: 随机种子
        """
        self.memory = deque(maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """添加一条经验 (s, a, s', r, done)"""
        self.memory.append(transition)

    def sample(self, sample_size):
        """随机采样指定数量的经验"""
        return random.sample(self.memory, sample_size)

    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.memory)


def SAC_train():
    # === 超参数设置 ===
    learning_rate_actor = 0.0003  # Actor学习率
    learning_rate_critic = 0.001  # Critic学习率
    learning_rate_alpha = 0.0003  # 温度参数α的学习率

    discount_factor_g = 0.99  # 折扣因子 γ
    hidden_dim = 128  # 隐藏层维度
    buffer_size = 100000  # 经验回放缓冲区大小
    batch_size = 128  # 批量大小
    tau = 0.005  # 软更新系数
    target_entropy = -1.0  # 目标熵，设置为 -dim(A)，Pendulum中动作维度为1

    # === 初始化环境和网络 ===
    env = gym.make('Pendulum-v1')
    states_dim = env.observation_space.shape[0]  # 状态维度
    actions_dim = env.action_space.shape[0]  # 动作维度
    action_bound = env.action_space.high[0]  # 动作范围上界

    actor = PolicyNet(states_dim, actions_dim, hidden_dim, action_bound).to(device)
    critic = TwinQValueNet(states_dim, actions_dim, hidden_dim).to(device)
    critic_target = TwinQValueNet(states_dim, actions_dim, hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())  # 初始化目标网络

    # === 温度参数 α ===
    log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)  # 初始化log(α)
    alpha_optim = torch.optim.Adam([log_alpha], lr=learning_rate_alpha)

    # === 优化器 ===
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate_critic)

    replay_buffer = ReplayMemory(buffer_size)
    rewards_all_episodes = []
    episodes_iter = 0

    # === 训练主循环 ===
    for episode in range(200):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32, device=device)
        rewards_one_episode = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                action, _ = actor.sample(state)  # 从策略中采样动作
                action = action.cpu().numpy().flatten()

            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards_one_episode += reward

            # 存储经验到缓冲区
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            done_tensor = torch.tensor([terminated], dtype=torch.float32, device=device)
            actions = torch.tensor(action, dtype=torch.float32, device=device)
            replay_buffer.append((state, actions, next_state_tensor, reward_tensor, done_tensor))
            state = next_state_tensor

            # === 训练网络 ===
            if len(replay_buffer) > batch_size:
                # 采样批量数据
                mini_batch = replay_buffer.sample(batch_size)
                states, actions, new_states, rewards, terminations = zip(*mini_batch)

                states = torch.stack(states)
                actions = torch.stack(actions)
                new_states = torch.stack(new_states)
                rewards = torch.stack(rewards)
                terminations = torch.stack(terminations)

                # === 更新Critic ===
                # 目标Q值公式：y = r + γ * (1 - d) * [min(Q1', Q2')(s', a') - α * log π(a'|s')]
                with torch.no_grad():
                    next_actions, next_log_probs = actor.sample(new_states)
                    target_q1, target_q2 = critic_target(new_states, next_actions)
                    target_q = torch.min(target_q1, target_q2) - log_alpha.exp() * next_log_probs
                    q_targets = rewards + discount_factor_g * (1 - terminations) * target_q

                current_q1, current_q2 = critic(states, actions)
                critic_loss = F.mse_loss(current_q1, q_targets) + F.mse_loss(current_q2, q_targets)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)  # 梯度裁剪
                critic_optimizer.step()

                # === 更新Actor ===
                # Actor损失函数：L = α * log π(a|s) - min(Q1, Q2)(s, a)
                actions_pred, log_probs = actor.sample(states)
                q1_pred, q2_pred = critic(states, actions_pred)
                actor_loss = (log_alpha.exp().detach() * log_probs - torch.min(q1_pred, q2_pred)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)  # 梯度裁剪
                actor_optimizer.step()

                # === 更新温度参数 α ===
                # 目标：使熵 H(π) 接近目标熵，损失函数：L(α) = -α * (log π(a|s) + H_target)
                alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                # === 软更新目标网络 ===
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        rewards_all_episodes.append(rewards_one_episode)
        episodes_iter += 1

        if episodes_iter % 20 == 0:
            mean_reward = np.mean(rewards_all_episodes[-20:])
            print(f'回合: {episodes_iter}, 平均奖励: {mean_reward:.3f}, Alpha: {log_alpha.exp().item():.4f}')

            plt.plot(rewards_all_episodes)
            plt.savefig('SAC-Pendulum_train.png')
            plt.close()

            torch.save(actor.state_dict(), "SAC-Pendulum_actor.pt")
            #torch.save(critic.state_dict(), "SAC-Pendulum_critic.pt")


def SAC_test():
    env = gym.make('Pendulum-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    actor = PolicyNet(state_dim, action_dim, 128, action_bound).to(device)
    if os.path.exists('SAC-Pendulum_actor.pt'):
        actor.load_state_dict(torch.load('SAC-Pendulum_actor.pt', map_location=device))
    else:
        print("SAC-Pendulum_actor.pt 不存在")
        return
    actor.eval()

    for _ in range(5):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action, _ = actor.sample(state_tensor)
                action = action.cpu().numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f"测试回合奖励: {total_reward:.1f}")
    os.system('pause')
    env.close()

# ================== BC网络结构 ==================
class BCNetwork(nn.Module):
    """行为克隆专用网络"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, state):
        return self.net(state)

def collect_expert_data():
    """使用训练好的策略收集专家数据"""
    num_episodes = 100
    max_steps = 200
    env = gym.make('Pendulum-v1')
    expert_buffer = deque(maxlen = num_episodes * max_steps)

    # 加载训练好的模型
    actor = PolicyNet(state_dim=3, action_dim=1, hidden_dim=128, action_bound=2.0).to(device)
    if os.path.exists('SAC-Pendulum_actor.pt'):
        actor.load_state_dict(torch.load('SAC-Pendulum_actor.pt', map_location=device))
    else:
        print("SAC-Pendulum_actor.pt 不存在")
        return expert_buffer
    actor.eval()

    for _ in range(num_episodes):
        state = env.reset()[0]
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).to(device)
            action, _ = actor.sample(state_tensor)
            action = action.detach().numpy().flatten()

            expert_buffer.append((state.copy(), action.copy()))

            next_state, _, done, _, _ = env.step(action)
            state = next_state
            if done:
                break
    env.close()
    return expert_buffer

# ================== BC训练函数 ==================
def train_bc(expert_data):
    # 定义行为克隆网络的隐藏层维度为128
    bc_hidden_dim = 128
    # 设置学习率为0.001（1e-3），控制参数更新的步长
    lr = 1e-3
    # 设置训练的总轮数为100轮
    epochs = 100
    # 设置每个批次的数据量为64
    batch_size = 64

    """训练行为克隆网络"""
    # 初始化行为克隆网络，输入状态维度为3，输出动作维度为1，隐藏层维度为128
    # .to(device) 将网络移动到指定设备（例如CPU或GPU）
    bc_net = BCNetwork(state_dim=3, action_dim=1, hidden_dim=bc_hidden_dim).to(device)
    # 使用Adam优化器优化网络参数，学习率为lr
    optimizer = torch.optim.Adam(bc_net.parameters(), lr=lr)
    # 定义损失函数为均方误差（MSE），用于衡量预测动作与真实动作的差距
    criterion = nn.MSELoss()

    # 从expert_data中提取所有状态（x[0]）并转换为numpy数组，再转为PyTorch浮点张量
    # .to(device) 将数据移动到指定设备
    states = torch.FloatTensor(np.array([x[0] for x in expert_data])).to(device)
    # 从expert_data中提取所有动作（x[1]）并转换为numpy数组，再转为PyTorch浮点张量
    # .to(device) 将数据移动到指定设备
    actions = torch.FloatTensor(np.array([x[1] for x in expert_data])).to(device)

    # 初始化一个空列表，用于存储每轮训练的总损失
    losses = []
    # 开始训练循环，遍历指定的epochs轮数
    for epoch in range(epochs):
        # 生成一个随机打乱的索引序列，长度为states的总数，用于随机抽取批次
        permutation = torch.randperm(len(states))
        # 按批次大小遍历整个数据集，i是每个批次的起始索引
        for i in range(0, len(states), batch_size):
            # 从permutation中提取从索引i到i+batch_size的子序列，作为当前批次的索引
            # 如果剩余数据不足batch_size，则取剩余所有数据
            indices = permutation[i:i + batch_size]
            # 根据indices从states中提取当前批次的状态数据
            batch_states = states[indices]
            # 根据indices从actions中提取当前批次的动作数据
            batch_actions = actions[indices]

            # 通过网络预测当前批次状态对应的动作
            pred_actions = bc_net(batch_states)
            # 计算预测动作与真实动作之间的均方误差损失
            loss = criterion(pred_actions, batch_actions)

            # 清空优化器中的梯度，防止累积之前计算的梯度
            optimizer.zero_grad()
            # 反向传播，计算损失对网络参数的梯度
            loss.backward()
            # 根据计算的梯度更新网络参数
            optimizer.step()

        # 在不计算梯度的情况下评估整个数据集的损失
        with torch.no_grad():
            # 使用训练后的网络预测所有状态的动作，并计算与真实动作的总损失
            total_loss = criterion(bc_net(states), actions).item()
        # 将当前轮次的总损失添加到losses列表中
        losses.append(total_loss)
        # 打印当前轮次的训练进度和损失值，保留4位小数
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # 返回训练完成的网络模型
    return bc_net

# ================== BC测试函数 ==================
def test_bc(bc_net, num_episodes=5):
    """测试行为克隆网络"""
    env = gym.make('Pendulum-v1', render_mode='human')
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = bc_net(state_tensor).cpu().numpy()[0]

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        print(f"Episode Reward: {episode_reward:.1f}")
    os.system('pause')
    env.close()
    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(total_rewards):.1f}")
    return total_rewards

if __name__ == '__main__':

    #SAC_train()
    #SAC_test()

    print("\nTraining BC network...")
    expert_data = collect_expert_data()

    print("\nTraining BC network...")
    bc_model = train_bc(expert_data)

    print("\nTesting BC network...")
    test_bc(bc_model, num_episodes=5)