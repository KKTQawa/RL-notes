#!/usr/bin/env python3
import os
import yaml
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
from collections import deque, OrderedDict
from types import SimpleNamespace as SN
import gymnasium as gym
import rware


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def mlp_block(in_dim, out_dim, activation=None, device=None):
    block = [nn.Linear(in_dim, out_dim, device=device)]
    if activation is not None:
        block.append(activation())
    return block


def space2shape(space):
    if space is None:
        return None
    if isinstance(space, dict):
        return {k: space2shape(v) for k, v in space.items()}
    return space.shape

# ============ Replay Buffer ============
class MARLReplayBuffer:
    def __init__(self, buffer_size, batch_size, n_agents, obs_dim, act_dim, state_dim):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0
        self.clear()

    def clear(self):
        shp = (self.buffer_size,)
        self.obs = np.zeros(shp + (self.n_agents, self.obs_dim), dtype=np.float32)
        self.obs_next = np.zeros(shp + (self.n_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(shp + (self.n_agents,), dtype=np.int64)
        self.rewards = np.zeros(shp + (self.n_agents,), dtype=np.float32)
        self.terminals = np.zeros(shp + (self.n_agents,), dtype=np.bool_)
        self.state = np.zeros(shp + (self.state_dim,), dtype=np.float32)
        self.state_next = np.zeros(shp + (self.state_dim,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, obs, actions, rewards, terminals, obs_next, state, state_next):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.terminals[self.ptr] = terminals
        self.obs_next[self.ptr] = obs_next
        self.state[self.ptr] = state
        self.state_next[self.ptr] = state_next
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        bs = self.batch_size if batch_size is None else batch_size
        step_choices = np.random.choice(self.size, bs)
        return {
            'obs': self.obs[step_choices],
            'actions': self.actions[step_choices],
            'rewards': self.rewards[step_choices],
            'terminals': self.terminals[step_choices],
            'obs_next': self.obs_next[step_choices],
            'state': self.state[step_choices],
            'state_next': self.state_next[step_choices],
            'batch_size': bs,
        }

# ============ Representation ============
class BasicMLP(nn.Module):
    def __init__(self, input_shape, hidden_sizes, activation=nn.ReLU, device=None):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        layers = []
        inp = input_shape[0]
        for h in hidden_sizes:
            layers.extend(mlp_block(inp, h, activation, device))
            inp = h
        self.model = nn.Sequential(*layers)

    def forward(self, observations):
        t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(t)}

# ============ Q Head ============
class BasicQhead(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_sizes, activation=nn.ReLU, device=None):
        super().__init__()
        layers = []
        inp = state_dim
        for h in hidden_sizes:
            layers.extend(mlp_block(inp, h, activation, device))
            inp = h
        layers.extend(mlp_block(inp, n_actions, None, device))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ============ QMIX Mixer ============
class MixingNet(nn.Module):
    def __init__(self, dim_state, dim_hidden, dim_hypernet_hidden, n_agents, device=None):
        super().__init__()
        self.device = device
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.n_agents = n_agents
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(dim_state, dim_hypernet_hidden),
            nn.ReLU(),
            nn.Linear(dim_hypernet_hidden, dim_hidden * n_agents)
        ).to(device)
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(dim_state, dim_hypernet_hidden),
            nn.ReLU(),
            nn.Linear(dim_hypernet_hidden, dim_hidden)
        ).to(device)
        self.hyper_b_1 = nn.Linear(dim_state, dim_hidden).to(device)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(dim_state, dim_hypernet_hidden),
            nn.ReLU(),
            nn.Linear(dim_hypernet_hidden, 1)
        ).to(device)

    def forward(self, values_n, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states = states.reshape(-1, self.dim_state)
        agent_qs = values_n.reshape(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.dim_hidden)
        b1 = self.hyper_b_1(states).view(-1, 1, self.dim_hidden)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        w2 = torch.abs(self.hyper_w_2(states)).view(-1, self.dim_hidden, 1)
        b2 = self.hyper_b_2(states).view(-1, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(-1, 1)

# ============ QMIX_policy (policy) ============
class QMIX_policy(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions, dim_state,
                 repr_hidden, q_hidden, mixer_hidden, hyper_hidden,
                 activation=nn.ReLU, device=None):
        super().__init__()
        self.device = device
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.representation = BasicMLP(
            input_shape=(obs_dim + n_agents,),  # + agent_id for parameter sharing
            hidden_sizes=repr_hidden,
            activation=activation,
            device=device
        )
        self.target_representation = deepcopy(self.representation)

        repr_out = self.representation.output_shapes['state'][0]
        self.eval_qhead = BasicQhead(repr_out, n_actions, q_hidden, activation, device)
        self.target_qhead = deepcopy(self.eval_qhead)

        self.eval_mixer = MixingNet(dim_state, mixer_hidden, hyper_hidden, n_agents, device)
        self.target_mixer = deepcopy(self.eval_mixer)

    def forward(self, obs, agent_ids):
        obs_concat = torch.cat([obs, agent_ids], dim=-1)
        rep = self.representation(obs_concat)
        q = self.eval_qhead(rep['state'])
        return q

    def target_q(self, obs, agent_ids):
        obs_concat = torch.cat([obs, agent_ids], dim=-1)
        rep = self.target_representation(obs_concat)
        q = self.target_qhead(rep['state'])
        return q

    def q_tot(self, individual_qs, states):
        return self.eval_mixer(individual_qs, states)

    def target_q_tot(self, individual_qs, states):
        return self.target_mixer(individual_qs, states)

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_qhead.parameters(), self.target_qhead.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_mixer.parameters(), self.target_mixer.parameters()):
            tp.data.copy_(ep)

# ============ QMIX Agent ============
class QMIXAgent:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.n_agents = config.n_agents
        self.obs_dim = config.obs_dim
        self.act_dim = config.act_dim
        self.state_dim = config.state_dim

        self.policy = QMIX_policy(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_actions=self.act_dim,
            dim_state=self.state_dim,

            repr_hidden=config.representation_hidden_size,
            q_hidden=config.q_hidden_size,
            mixer_hidden=config.hidden_dim_mixing_net,
            hyper_hidden=config.hidden_dim_hyper_net,

            activation=nn.ReLU,
            device=self.device
        ).to(self.device)

        self.memory = MARLReplayBuffer(
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), config.learning_rate, eps=1e-5)
        total_iters = (config.running_steps - config.start_training) // config.training_frequency
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=total_iters
        )
        self.mse_loss = nn.MSELoss()
        self.iterations = 0

        self.e_greedy = config.start_greedy
        self.delta_egreedy = (config.start_greedy - config.end_greedy) / max(1, config.decay_step_greedy)
        self.double_q = config.double_q
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.start_training = config.start_training
        self.training_frequency = config.training_frequency
        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm

    @torch.no_grad()
    def get_actions(self, obs_np, epsilon=None):
        eps = epsilon if epsilon is not None else self.e_greedy
        bs = obs_np.shape[0]
        agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

        q_values = self.policy(obs_t, agent_ids)
        actions = q_values.argmax(dim=-1)

        if np.random.random() < eps:
            actions = torch.randint(0, self.act_dim, (bs, self.n_agents), device=self.device)

        return actions.cpu().numpy(), q_values.cpu().numpy()

    def store_experience(self, obs, actions, rewards, terminals, obs_next, state, state_next):
        self.memory.store(obs, actions, rewards, terminals, obs_next, state, state_next)

    def train_epoch(self):
        if self.iterations > 0 and self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        sample = self.memory.sample()
        bs = sample['batch_size']
        agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)

        obs = torch.as_tensor(sample['obs'], dtype=torch.float32, device=self.device)
        obs_next = torch.as_tensor(sample['obs_next'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(sample['actions'], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(sample['rewards'], dtype=torch.float32, device=self.device)
        terminals = torch.as_tensor(sample['terminals'], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(sample['state'], dtype=torch.float32, device=self.device)
        state_next = torch.as_tensor(sample['state_next'], dtype=torch.float32, device=self.device)

        rewards_tot = rewards.mean(dim=-1, keepdim=True)
        terminals_tot = terminals.all(dim=-1, keepdim=True).float()

        q_eval_all = self.policy(obs, agent_ids)
        q_eval_chosen = q_eval_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_next_all = self.policy.target_q(obs_next, agent_ids)
            if self.double_q:
                next_actions = self.policy(obs_next, agent_ids).argmax(dim=-1, keepdim=True)
                q_next_chosen = q_next_all.gather(-1, next_actions).squeeze(-1)
            else:
                q_next_chosen = q_next_all.max(dim=-1).values

        q_tot_eval = self.policy.q_tot(q_eval_chosen, state)
        q_tot_next = self.policy.target_q_tot(q_next_chosen, state_next)
        q_tot_target = rewards_tot + (1 - terminals_tot) * self.gamma * q_tot_next

        loss = self.mse_loss(q_tot_eval, q_tot_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        self.iterations += 1
        return loss.item(), q_tot_eval.mean().item()

    def update_epsilon(self):
        self.e_greedy = max(self.config.end_greedy, self.e_greedy - self.delta_egreedy)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


# ============ Robotic Warehouse Env Wrapper ============
class RoboticWarehouseEnv:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env_id, render_mode=config.render_mode,
                            max_steps=config.max_episode_steps)
        self.n_agents = len(self.env.action_space)
        self.agents = [f'agent_{i}' for i in range(self.n_agents)]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_space = self.env.observation_space[0]
        self.max_episode_steps = config.max_episode_steps
        self._seeded = False

    def reset(self):
        if not self._seeded:
            obs, info = self.env.reset(seed=self.config.env_seed)
            self._seeded = True
        else:
            obs, info = self.env.reset()
        self._step = 0
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions.tolist())
        self._step += 1
        return np.array(obs, dtype=np.float32), np.array(reward, dtype=np.float32), terminated, truncated, info

    def close(self):
        self.env.close()

def get_global_state(obs):
    return obs.reshape(-1)

def draw(ep_total_rewards, ep_avg_rewards, ep_losses, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(ep_total_rewards, color='tab:blue')
    axes[0].set_ylabel('Episode Total Reward')
    axes[0].set_xlabel('Episode')
    axes[0].grid(True)

    axes[1].plot(ep_avg_rewards, color='tab:orange')
    axes[1].set_ylabel('Average Reward (per step)')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True)

    axes[2].plot(ep_losses, color='tab:green')
    axes[2].set_ylabel('Average Loss')
    axes[2].set_xlabel('Episode')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")

def train(config):
    env = RoboticWarehouseEnv(config)
    agent = QMIXAgent(config)
    n_agents = config.n_agents

    obs = env.reset()
    #print("obs:", obs)
    state = get_global_state(obs)

    episode_total_reward = 0.0
    episode_steps = 0
    total_steps = 0
    train_episode = 0

    pbar = range(1, config.running_steps + 1)
    if config.use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(pbar)

    best_avg_reward = -1e9
    ep_total_rewards = []
    ep_avg_rewards = []
    ep_losses = []
    current_losses = []

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config.model_dir}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for step in pbar:
        obs_batch = obs[np.newaxis, :]
        actions, _ = agent.get_actions(obs_batch)
        o, r, d, tr, _ = env.step(actions[0])
        done = d or tr

        terminal = np.full(n_agents, done)
        obs_next = o
        state_next = get_global_state(obs_next)

        agent.store_experience(obs, actions[0], r, terminal, obs_next, state, state_next)

        episode_total_reward += r.sum()
        episode_steps += 1

        obs = obs_next
        state = state_next
        total_steps += 1

        if total_steps >= config.start_training and step % config.training_frequency == 0:
            loss, q = agent.train_epoch()
            current_losses.append(loss)

        agent.update_epsilon()

        if done or episode_steps >= config.max_episode_steps:
            train_episode += 1
            avg_r = episode_total_reward / max(1, episode_steps)
            avg_loss = np.mean(current_losses) if current_losses else 0.0

            ep_total_rewards.append(episode_total_reward)
            ep_avg_rewards.append(avg_r)
            ep_losses.append(avg_loss)

            if avg_r > best_avg_reward:
                best_avg_reward = avg_r

            if config.use_tqdm:
                pbar.set_description(f"Ep {train_episode} R={episode_total_reward:.2f} avgR={avg_r:.2f} Avg_Loss: {avg_loss:.4f} Steps: {episode_steps}")

            episode_total_reward = 0.0
            episode_steps = 0
            current_losses = []
            obs = env.reset()
            state = get_global_state(obs)

            #print(f"Episode:{len(ep_total_rewards)} | Avg_R: {avg_r:.4f} | Avg_Loss: {avg_loss:.4f}")

        if step % config.eval_interval == 0 and step > 0:
            eval_reward = evaluate(config, agent)
            print(f"Evaluation at step {step}: avg_reward = {eval_reward:.4f}")
            agent.save_model(f"{save_dir}/{best_avg_reward:.4f}_step_{step}.pth")

    env.close()

    agent.save_model(f"{save_dir}/{best_avg_reward:.4f}_final.pth")
    print(f"Training completed. Model saved to {save_dir}")
    draw(ep_total_rewards, ep_avg_rewards, ep_losses, save_dir)


@torch.no_grad()
def evaluate(config, agent=None):
    if agent is None:
        agent = QMIXAgent(config)
        agent.load_model(f"{config.model_dir}/final.pth")
    print(f"Evaluation mode: {config.render_mode}")
    env = RoboticWarehouseEnv(config)
    ep_rewards = []
    for ep in range(config.test_episode):
        obs = env.reset()
        env.env.render()
        done = False
        ep_r = 0
        steps = 0
        while not done and steps < config.max_episode_steps:
            obs_t = obs.reshape(1, config.n_agents, config.obs_dim)
            agent_ids = torch.eye(config.n_agents, device=config.device).unsqueeze(0)
            q = agent.policy(
                torch.as_tensor(obs_t, dtype=torch.float32, device=config.device),
                agent_ids
            )
            #使用cpu进行评估
            actions = q.argmax(dim=-1).squeeze(0).cpu().numpy()
            #print(f"actions: {actions}")
            obs, reward, terminated, truncated, _ = env.step(actions)
            #print(f"reward: {reward}")#[0. 0.]
            #print(f"obs: {obs}")
            done = terminated or truncated
            ep_r += reward.sum()
            steps += 1
            env.env.render()
            #os.system("pause")
        ep_rewards.append(ep_r)
    env.close()
    return np.mean(ep_rewards)


def main():
    # parse args from yaml
    config_path = os.path.join(os.path.dirname(__file__), 'rware-tiny-2ag-v2.yaml')
    #basic_path = os.path.join(os.path.dirname(__file__), 'basic.yaml')

    algo_config = load_yaml(config_path)
    # basic_config = load_yaml(basic_path)

    # # merge
    # full_config = deepcopy(basic_config)
    # full_config.update(algo_config)

    config = SN(**algo_config)

    # device
    config.device = getattr(config, 'device', 'cpu')
    if config.device == 'cpu':
        config.device = 'cpu'
    elif 'cuda' in config.device:
        config.device = config.device if torch.cuda.is_available() else 'cpu'

    # build a single env to get dimensions
    temp_env = gym.make(config.env_id)
    config.n_agents = len(temp_env.action_space)
    config.obs_dim = temp_env.observation_space[0].shape[0]
    config.act_dim = temp_env.action_space[0].n
    config.state_dim = config.obs_dim * config.n_agents
    config.use_tqdm = True
    temp_env.close()

    print(f"QMIX on {config.env_id}")
    print(f"  Agents: {config.n_agents}, ObsDim: {config.obs_dim}, ActDim: {config.act_dim}, StateDim: {config.state_dim}")
    print(f"  Device: {config.device}")
    print(f"  Running steps: {config.running_steps}")
    print(f"  Buffer size: {config.buffer_size}, Batch size: {config.batch_size}")

    mode = getattr(config, 'mode', 'train')
    if mode == 'train':
        train(config)
    else:
        eval_reward = evaluate(config)
        print(f"Evaluation: avg_reward = {eval_reward:.4f}")


if __name__ == '__main__':
    main()
