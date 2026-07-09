from pettingzoo.butterfly import pistonball_v6
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import cv2

EPISODES = 500
BUFFER_SIZE = 10000
BATCH_SIZE = 32
START_SIZE = 1000
GAMMA = 0.99
TARGET_UPDATE = 200
MAX_STEP = 130

ACTION_LIST = np.linspace(-1.0, 1.0, 11)  # 11个离散动作
print("ACTION_LIST:", ACTION_LIST)
N_ACTIONS = len(ACTION_LIST)
# def preprocess_obs(obs):
#     # (H, W, C) 
#     obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#     obs = cv2.resize(obs, (316, 84))
#     obs = obs.astype(np.float32) / 255.0
#     return obs
def preprocess_obs(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (316, 84))
    return obs.astype(np.uint8)
class QNet(nn.Module):
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # (84,316) -> downsample
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 这里用 dummy forward 自动算 flatten 维度（推荐）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 84, 316)
            n_flatten = self.cnn(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # x: (B, 84, 316)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, 84, 316)

        x = self.cnn(x)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action_idxs, reward, next_state, done):
        """
        state      (N,H,W)
        action     (N,)
        reward     float
        next_state (N,H,W)
        done       bool
        """
        self.buffer.append((
            state.astype(np.float32),
            action_idxs.astype(np.uint8, copy=False),
            float(reward),
            next_state.astype(np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, action_idxs, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),        # (B,N,H,W)
            np.stack(action_idxs),       # (B,N)
            np.array(rewards, dtype=np.float32),      # (B,)
            np.stack(next_states),   # (B,N,H,W)
            np.array(dones, dtype=np.float32)         # (B,)
        )

    def __len__(self):
        return len(self.buffer)

class VDN:
    def __init__(self, n_actions=N_ACTIONS, device='cpu'):
        self.device=device
        self.q = QNet(n_actions).to(device)
        self.target_q = QNet(n_actions).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.n_actions=n_actions

        self.optim = optim.Adam(self.q.parameters(), lr=1e-4)

    def forward(self, obs, eps=0.1):
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1) 

        obs = torch.FloatTensor(obs).float().div_(255.0).unsqueeze(0).to(self.device)
        q = self.q(obs).detach().cpu().numpy()[0]# 移回CPU进行numpy操作
        return np.argmax(q).item()
    
    def act(self, obs):
        obs = torch.FloatTensor(obs).float().div_(255.0).unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs).detach().numpy()[0]
        return q.argmax(1).item()

    def update_target(self):
        self.target_q.load_state_dict(self.q.state_dict())

def train():
    env = pistonball_v6.parallel_env(n_pistons=20)
    env.reset(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    shared_agent = VDN(N_ACTIONS,device)
    buffer = ReplayBuffer()
    losses = []

    update_step=0

    for ep in range(EPISODES):

        obs_dict, _ = env.reset(seed=ep)
        #print("obs_dict:", obs_dict)#{'piston_0':(457, 120, 3),'piston_1':(),...}
        done = False
        step = 0

        episode_reward = 0
        episode_loss = 0

        while (not done) and step < MAX_STEP:
            #print("env.agents:", env.agents)

            step += 1
            obs = np.stack([
                preprocess_obs(obs_dict[a])
                for a in env.agents
            ])                       # (N,H,W)

            action_idxs=np.array([shared_agent.forward(o, eps) for o in obs])#(N,)

            action_dict = {
                agent: np.array([ACTION_LIST[action_idxs[i]]], dtype=np.float32)
                for i, agent in enumerate(env.agents)
            }

            next_obs_dict, rewards, terms, truncs, infos = env.step(action_dict)

            #print("rewards:", rewards)#dict
            reward = np.mean(list(rewards.values()))
            done = all(terms.values()) or all(truncs.values())

            if done:
                break

            next_obs = np.stack([
                preprocess_obs(next_obs_dict[a])
                for a in env.agents
            ])

            buffer.add(
                obs,
                action_idxs,
                reward,
                next_obs,
                done
            )
            if buffer.__len__() ==10000:
                print("buffer.__len__:", 10000)

            obs_dict = next_obs_dict
            episode_reward += reward

            if len(buffer) >= START_SIZE:
                update_step += 1

                states, action_idxs, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                B = states.shape[0]
                N = states.shape[1]

                state_tensor = torch.FloatTensor(states).float().div_(255.0).view(
                    B * N, 84, 316
                ).to(device)#B*N=num_samples

                next_state_tensor = torch.FloatTensor(next_states).float().div_(255.0).view(
                    B * N, 84, 316
                ).to(device)

                action_idx_tensor = torch.LongTensor(action_idxs).view(
                    B * N, 1
                ).to(device)#张量索引必须使用LongTensor类型
                rewards = torch.FloatTensor(rewards).to(device)#numpy.ndarray->torch.FloatTensor
                dones = torch.FloatTensor(dones).to(device)#numpy.ndarray->torch.FloatTensor

                q = shared_agent.q(state_tensor)#(B*N, action_dim)=(B*N,11)
                q = q.gather(1, action_idx_tensor).view(B, N)#(B*N,11)->(B*N, 1)->(B,N)
                q_total = q.sum(dim=1)#(B,)

                with torch.no_grad():

                    next_q = shared_agent.target_q(next_state_tensor)#(B*N, action_dim)=(B*N,11)
                    next_q = next_q.view(B, N,-1)#(B*N,11)->(B,N,11)
                    next_q = next_q.max(2)[0]#(B,N,11)->(B*N,)
                    
                    target_total = rewards +  GAMMA * (1 - dones) * next_q.sum(dim=1)#(B,)
                
                loss = F.mse_loss(q_total, target_total)
                shared_agent.optim.zero_grad()

                loss.backward()

                shared_agent.optim.step()
                episode_loss += loss.item()

                if update_step % TARGET_UPDATE == 0:
                    #print("Start update target")
                    shared_agent.update_target()

        eps = max(eps_min, eps * eps_decay)
        losses.append(episode_loss)

        print(
            f"Episode {ep:4d} "
            f"Step {step:4d} "
            f"Reward {episode_reward:.2f} "
            f"Loss {episode_loss:.4f}"
        )

    env.close()

    return shared_agent, losses
def test(agent):
    env = pistonball_v6.parallel_env(n_pistons=20, render_mode="human") 
    
    for ep in range(2):
        obs_dict, _ = env.reset(seed=ep)
        done = False
        step = 0
        episode_reward = 0
        
        print(f"\n=== Episode {ep} ===")
        
        while (not done) and step < MAX_STEP: 
            step += 1

            obs = np.stack([
                preprocess_obs(obs_dict[a])
                for a in env.agents
            ])  

            action_idxs = np.array([
                agent.act(o)  
                for o in obs
            ])

            action_dict = {
                agent_name: np.array([ACTION_LIST[action_idxs[i]]], dtype=np.float32)
                for i, agent_name in enumerate(env.agents)
            }

            next_obs_dict, rewards, terms, truncs, infos = env.step(action_dict)
            
            reward = np.mean(list(rewards.values()))
            done = all(terms.values()) or all(truncs.values())
            
            obs_dict = next_obs_dict
            episode_reward += reward
            print("reward:",reward)
            os.system("pause")
        
        print(f"Episode {ep} finished: Total Reward = {episode_reward:.2f}, Steps = {step}")
        os.system("pause")
    
    env.close()

def run_env(env_id):
    try:
        env = pistonball_v6.env( render_mode="human")
        env.reset(seed=42)

        print(f"Running environment: {env_id}")

        ep=0
        for agent in env.agent_iter():
            print("agent:",agent)
            ep+=1
            if ep==1:
                print('observation_space:',env.observation_space(agent))
                print('action_space:',env.action_space(agent))
                print("agent:",agent)

            observation, reward, termination, truncation, info = env.last()
            print("reward:",reward)
            #print("observation:",observation)

            if termination or truncation:
                action = None
            else:
                #action = env.action_space(agent).sample()
                action = np.array([float(input(f"请输入{agent}的动作(-1,1):"))], dtype=np.float32)

            env.step(action)

    except KeyboardInterrupt:
        print("\nExiting:", env_id)

    env.close()

if __name__ == "__main__":
    env_id="pistonball_v6"
    #run_env(env_id)
    agents, losses = train()
    plt.plot(losses)
    plt.title("VDN_pistonball_v6_loss")
    plt.savefig("VDN_pistonball_v6_loss.png")
    plt.show()
    os.system("pause")
    test(agents)
    
