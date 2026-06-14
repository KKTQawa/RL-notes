import gymnasium as gym, torch, numpy as np
from torch import nn, optim
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import os

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity

        self.buffer = []
        self.position = 0

    @property
    def size(self):
        """当前缓冲区中实际存储的样本数"""
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # 缓冲区未满
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        # 缓冲区已满，覆盖最旧数据
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device='cuda'):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones

    def clear(self):
        self.buffer.clear()
        self.position = 0

class Actor(nn.Module):
    #A网络
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )


    def forward(self, x):
        return self.net(x)
class Critic(nn.Module):
    #Q网络
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.Q1(torch.cat([state, action], dim=1)), self.Q2(torch.cat([state, action], dim=1))
class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, device='cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu', replay_buffer_capacity=10000):
        
        #共5个网络
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.tau = tau# 目标网络的软更新参数

        self.device = device
        self.target_entropy = -action_dim#这里选择-action_dim是基于经验
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_buffer_capacity)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.log_std_min = -20
        self.log_std_max = 2

        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)#参与计算图
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    # 采样动作
    def act(self, obs, evaluate=False):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        pred = self.actor(obs)#[batch_size, action_dim*2](action_mean, action_log_std)均值和标准差
        action_mean, action_log_std = torch.chunk(pred, 2, dim=-1)#拆分出来
        if evaluate:
            return torch.tanh(action_mean), None
        log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(action_mean, std)
        normal_sample = dist.rsample()#高斯分布重采样

        action = torch.tanh(normal_sample)

        #计算log\pi(a)
        log_prob = dist.log_prob(normal_sample)
        correction = 2. * (np.log(2.) - normal_sample - F.softplus(-2. * normal_sample))
        log_prob -= correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob#a,log\pi(a)

    def train(self, batch_size = 256):
        if self.replay_buffer.size < batch_size: return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        #这里目标网络不反向传播参数进行更新，其参数要等后面抄袭
        with torch.no_grad():
            next_actions, new_log_prob = self.act(next_states)
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            y = rewards + (1 - dones) * self.gamma * (target_Q - self.alpha.item() * new_log_prob)#item()是为了将tensor转换为标量，不参与反向传播。这里是为了让/alpha与target_loss无关
        curr_Q1, curr_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(curr_Q1, y) + F.mse_loss(curr_Q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.update_target()

        new_actions, log_prob = self.act(states)#现在重新预测动作
        q1, q2 = self.critic(states, new_actions)
        q_min = torch.min(q1, q2)
        actor_loss = (self.alpha.item() * log_prob - q_min).mean()#Actor损失函数

        self.actor_optimizer.zero_grad()
        actor_loss.backward()#参与actor_loss计算图的所有参数的梯度都被计算了
        self.actor_optimizer.step()#只有actor网络的参数被更新了

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def update_target(self):
        #以tau的频率抄袭原始网络
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """
        保存所有状态，确保既能用于测试，也能用于恢复训练
        """
        torch.save({
            # --- 模型参数 (测试/推理 必须) ---
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),  # 恢复训练需要
            'log_alpha': self.log_alpha.detach(),  # 恢复训练需要

            # --- 优化器状态 (恢复训练 必须) ---
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, filename)

    def load(self, filename, evaluate=False):
        """
        加载模型
        :param filename: 模型路径
        :param evaluate:
               True  -> 仅加载 Actor 和 Critic (用于测试/验证)
               False -> 加载所有优化器和参数 (用于继续训练)
        """
        checkpoint = torch.load(filename, map_location=self.device)

        # 1. 加载网络参数 (无论训练还是测试都需要 Actor)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        # 2. 如果是测试模式，加载到这里就够了
        if evaluate:
            # 设为评估模式 (虽然 SAC Actor 通常没有 Dropout/BatchNorm，但这是好习惯)
            self.actor.eval()
            self.critic.eval()
            print(f"Loaded model from {filename} (Evaluation Mode)")
            return

        # 3. 如果是继续训练模式，必须加载优化器和目标网络
        self.target_critic.load_state_dict(checkpoint['target_critic'])

        # 恢复 log_alpha 的值 (关键！否则 Alpha 会重置)
        # 必须使用 .data.copy_ 来保持 requires_grad=True 的属性
        self.log_alpha.data.copy_(checkpoint['log_alpha'])

        # 加载优化器
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        # 恢复训练模式
        self.actor.train()
        self.critic.train()
        print(f"Loaded model from {filename} (Resume Training Mode)")

torch.set_float32_matmul_precision('high')

env = gym.make('HalfCheetah-v5', render_mode=None)
eval_env = gym.make('HalfCheetah-v5', render_mode=None)
run_env = gym.make('HalfCheetah-v5', render_mode='human')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

#重要可调超参数
model = SAC(state_dim, action_dim, hidden_dim=256, replay_buffer_capacity = 1000000)
episodes = 2500#大约2000之后出现新纪录 共计13个小时左右
warm_up = 18000

def evaluate_policy(agent, env, eval_episodes=5):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # 关键：evaluate=True (确定性策略，无噪声)
            with torch.no_grad():
                action, _ = agent.act(state, evaluate=True)
            action = action.detach().cpu().numpy()[0] * max_action
            state, reward, terminated, truncated, _ = env.step(action)
            avg_reward += reward
            done = terminated or truncated
    return avg_reward / eval_episodes

scores = []
eval_scores = []
train_interval = 4
base_score = 12000
pbar = tqdm(range(episodes), desc="Training")
step = 0
for episode in pbar:
    done = False
    state, _ = env.reset()
    score = 0
    while not done:
        step += 1
        if step <= warm_up:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = model.act(state)
            action = action.detach().cpu().numpy()[0] * max_action
        next_state, reward, termination, truncated, _ = env.step(action)
        done = termination or truncated
        score += reward
        model.store_transition(state, action, reward / 5.0, next_state, termination)
        state = next_state
        if step > warm_up and step % train_interval ==0:
            model.train()

    current_eval_score = 0
    # 最后一次评估
    if episode+1  == episodes:
        current_eval_score = evaluate_policy(model, run_env)
        eval_scores.append(current_eval_score)
        os.system('pause')
        model.save(f"model/Final-Half Cheetah-SAC.pth")
        tqdm.write(f"🔥 Final-Eval Score: {current_eval_score:.2f} (Saved)")
        os.system("pause")
            
    elif (episode + 1) % 20 == 0 and step >= warm_up:
        current_eval_score = evaluate_policy(model, eval_env)
        eval_scores.append(current_eval_score)

        # 保存最佳模型 (基于评估分数，而不是训练分数)
        if current_eval_score > base_score + 50:
            base_score = current_eval_score
            model.save(f"model/Half Cheetah-SAC-Best.pth")
            tqdm.write(f"🔥 New Best Eval Score: {base_score:.2f} (Saved)")

    scores.append(score)
    pbar.set_postfix(ep=episode, score=f"{score:.2f}", avg100=f"{np.mean(scores[-100:]):.2f}")

env.close()
eval_env.close()
run_env.close()

