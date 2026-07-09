from pettingzoo.classic import go_v5
import numpy as np
import os
import torch
import torch.nn as nn
import random
from collections import deque
import matplotlib.pyplot as plt

EPISODES=5000
BUFFER_SIZE=50000
START_SIZE=362*20
TARGET_UPDATE=80

class ValueNet(nn.Module):
    def __init__(self,n_actions=362):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(17, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(19 * 19 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # x: (B, 19, 19, 17)
        x = x.permute(0, 3, 1, 2).float()  # -> (B, 17, 19, 19)
        x = self.cnn(x)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, done, mask, mask2):
        self.buffer.append((s, a, r, s2, done, mask, mask2))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, m, m2 = map(np.array, zip(*batch))
        return s, a, r, s2, d, m, m2

    def __len__(self):
        return len(self.buffer)
    
class DQN:
    def __init__(self,  n_actions=362, lr=0.01, gamma=0.99):
        self.q = ValueNet( n_actions)
        self.target_q = ValueNet( n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.gamma = gamma
        self.n_actions = n_actions

    def forward(self, obs, mask, epsilon=0.1):
        if random.random() < epsilon:
            valid = np.where(mask == 1)[0]
            return np.random.choice(valid)

        obs = torch.FloatTensor(obs).unsqueeze(0)
        q = self.q(obs).detach().numpy()[0]

        q[mask == 0] = -1e9
        return int(np.argmax(q))
    
    def act(self, obs, mask):

        obs = torch.FloatTensor(obs).unsqueeze(0)
        q = self.q(obs).detach().numpy()[0]

        q[mask == 0] = -1e9
        return int(np.argmax(q))

    def update(self, batch):
        s, a, r, s2, d, m, m2 = batch

        s = torch.FloatTensor(s)
        s2 = torch.FloatTensor(s2)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        d = torch.FloatTensor(d)

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_q(s2)

            # mask invalid actions
            q_next[m2 == 0] = -1e9
            max_q_next = q_next.max(1)[0]

            target = r + self.gamma * (1 - d) * max_q_next

        loss = ((q_sa - target) ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def update_target(self):
        self.target_q.load_state_dict(self.q.state_dict())


def run_env(env_id):
    try:
        env = go_v5.env( render_mode="human")
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
                mask = observation["action_mask"]
                masks = np.where(mask == 1)#任意取出第一个元素
                print("合法动作索引:",masks)
                #action = np.random.choice(masks[0])
                action=int(input("请输入动作:"))

            env.step(action)

    except KeyboardInterrupt:
        print("\nExiting:", env_id)

    env.close()
def train(env_id):

    env = go_v5.env()
    env.reset(seed=42)

    agents = {}
    buffers = {}

    loss0 = []
    loss1 = []
    reward0 = []
    reward1 = []

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    update_step = 0

    for agent_id in env.agents:
        agents[agent_id] = DQN()
        buffers[agent_id] = ReplayBuffer()

    for ep in range(EPISODES):

        env.reset(seed=ep)

        ep_reward = {agent: 0.0 for agent in env.agents}
        ep_loss = {agent: 0.0 for agent in env.agents}
        counts = {agent: 0 for agent in env.agents}

        for agent in env.agent_iter():

            obs, reward, termination, truncation, info = env.last()
            mask = obs["action_mask"]
            ep_reward[agent] += reward
            done = termination or truncation

            if done:
                env.step(None)
                continue

            obs_state = obs["observation"].astype(np.float32)
            action = agents[agent].forward(obs_state, mask, eps)

            env.step(action)

            next_obs, next_reward, next_term, next_trunc, _ = env.last()
            next_mask = next_obs["action_mask"]
            next_state = next_obs["observation"].astype(np.float32)
            next_done = next_term or next_trunc

            buffers[agent].add(
                obs_state,
                action,
                reward,
                next_state,
                float(next_done),
                mask,
                next_mask
            )

            if len(buffers[agent]) > START_SIZE:
                batch = buffers[agent].sample(64)
                loss = agents[agent].update(batch)

                ep_loss[agent] += loss
                counts[agent] += 1

                update_step += 1

                if update_step % TARGET_UPDATE == 0:
                    agents[agent].update_target()

        eps = max(eps_min, eps * eps_decay)

        loss0.append(ep_loss["black_0"] / max(1, counts["black_0"]))#平均损失
        loss1.append(ep_loss["white_0"] / max(1, counts["white_0"]))

        reward0.append(ep_reward["black_0"])
        reward1.append(ep_reward["white_0"])

        if ep%100==0:
            print(f"Episode {ep} black_0 reward: {ep_reward['black_0']} loss: {ep_loss['black_0']} white_0 reward: {ep_reward['white_0']} loss: {ep_loss['white_0']}")

    env.close()

    return agents, loss0, loss1, reward0, reward1
def draw(env_id, loss1, loss2, reward1, reward2):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(loss1, label="Agent 0 Loss")
    plt.plot(loss2, label="Agent 1 Loss")
    plt.title(f"IQL_{env_id} Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(reward1, label="Agent 0 Reward")
    plt.plot(reward2, label="Agent 1 Reward")
    plt.title(f"IQL_{env_id} Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"IQL_{env_id}_train.png")
    plt.show()
    
def test(env_id, agents):
    env = go_v5.env(render_mode="human")
    print(f"Running environment: {env_id}")

    for ep in range(2):
        env.reset(seed=ep)

        ep_reward = {agent: 0.0 for agent in env.agents}

        for agent in env.agent_iter():

            obs, reward, termination, truncation, info = env.last()
            mask = obs["action_mask"]
            ep_reward[agent] += reward

            if termination or truncation:
                env.step(None)
                continue

            obs_state = obs["observation"].astype(np.float32)
            action = agents[agent].act(obs_state, mask)

            env.step(action)
            os.system("pause")

        print(f"Test Episode {ep}: {ep_reward}")
        os.system("pause")

    env.close()

if __name__ == "__main__":
    env_id="go_v5"
    #run_env(env_id)
    agents,loss1,loss2,reward1,reward2=train(env_id)
    draw(env_id,loss1,loss2,reward1,reward2)
    os.system("pause")
    test(env_id,agents)
    
