"""
Independent Q-Learning (IQL) for Robotic Warehouse (rware)
Based on xuance's IQL implementation, reimplemented without importing xuance.

Usage:
  python iql-rwarehouse.py --mode train          # train model
  python iql-rwarehouse.py --mode infer          # run inference
  python iql-rwarehouse.py --mode infer --render # render during inference
"""
import os
import sys
import time
import json
import math
import argparse
import numpy as np
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import gymnasium as gym

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


# ============================================================
# Config
# ============================================================
@dataclass
class IQLConfig:
    env_id: str = "rware-tiny-4ag-v2"
    env_seed: int = 1
    max_episode_steps: int = 100
    representation_hidden_size: List[int] = field(default_factory=lambda: [64])
    q_hidden_size: List[int] = field(default_factory=lambda: [64])
    seed: int = 42
    parallels: int = 8
    buffer_size: int = 50000
    batch_size: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99
    double_q: bool = True
    start_greedy: float = 1.0
    end_greedy: float = 0.05
    decay_step_greedy: int = 500000
    start_training: int = 1000
    running_steps: int = 2000000
    training_frequency: int = 1
    sync_frequency: int = 200
    use_grad_clip: bool = False
    grad_clip_norm: float = 0.5
    use_actions_mask: bool = True
    eval_interval: int = 50000
    test_episode: int = 5
    model_dir: str = "models/iql_rware/"
    device: str = "cpu"


# ============================================================
# MLP utility (same as xuance's mlp_block)
# ============================================================
def mlp_block(input_dim, output_dim, normalize=None, activation=None, initialize=None, device=None):
    layers = []
    linear = nn.Linear(input_dim, output_dim, device=device)
    if initialize is not None:
        initialize(linear.weight)
        nn.init.constant_(linear.bias, 0)
    layers.append(linear)
    if activation is not None:
        layers.append(activation())
    if normalize is not None:
        layers.append(normalize(output_dim, device=device))
    return layers


class Basic_MLP(nn.Module):
    """Representation network (same as xuance's Basic_MLP)."""
    def __init__(self, input_shape, hidden_sizes, normalize=None, initialize=None, activation=None, device=None):
        super().__init__()
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        layers = []
        input_dim = input_shape[0]
        for h in hidden_sizes:
            layers.extend(mlp_block(input_dim, h, normalize, activation, initialize, device))
            input_dim = h
        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, observations):
        return {'state': self.model(observations)}


class BasicQhead(nn.Module):
    """Q-network head (same as xuance's BasicQhead)."""
    def __init__(self, state_dim, n_actions, hidden_sizes, normalize=None, initialize=None, activation=None,
                 device=None):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.extend(mlp_block(input_dim, h, normalize, activation, initialize, device))
            input_dim = h
        layers.extend(mlp_block(input_dim, n_actions, None, None, initialize, device))
        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.model(x)


class BasicQnetwork(nn.Module):
    """
    Combined Q-network with eval and target networks (same as xuance's BasicQnetwork).
    Uses parameter sharing: single network for all agents with agent_id appended.
    """
    def __init__(self, obs_dim, n_agents, n_actions, representation_hidden_size=(64,), q_hidden_size=(64,),
                 activation=None, device=None):
        super().__init__()
        self.device = device
        self.n_agents = n_agents
        self.n_actions = n_actions
        act_fn = activation or nn.ReLU

        # Representation networks
        self.representation = Basic_MLP(
            input_shape=(obs_dim,), hidden_sizes=list(representation_hidden_size),
            activation=act_fn, device=device)
        self.target_representation = deepcopy(self.representation)

        # Q-head input dim = representation output + agent_id (one-hot)
        state_dim = representation_hidden_size[-1] + n_agents

        self.eval_Qhead = BasicQhead(state_dim, n_actions, list(q_hidden_size), activation=act_fn, device=device)
        self.target_Qhead = deepcopy(self.eval_Qhead)

    @property
    def parameters_model(self):
        return list(self.representation.parameters()) + list(self.eval_Qhead.parameters())

    def forward(self, obs_input, agent_ids, avail_actions=None):
        """
        Args:
            obs_input: [batch * n_agents, obs_dim]
            agent_ids: [batch * n_agents, n_agents]
            avail_actions: [batch * n_agents, n_actions] or None
        Returns: (None, argmax_actions, evalQ)
        """
        features = self.representation(obs_input)['state']
        q_input = torch.cat([features, agent_ids], dim=-1)
        q_values = self.eval_Qhead(q_input)

        if avail_actions is not None:
            q_masked = q_values.clone().detach()
            q_masked[avail_actions == 0] = -1e10
            argmax_actions = q_masked.argmax(dim=-1)
        else:
            argmax_actions = q_values.argmax(dim=-1)

        return None, argmax_actions, q_values

    def Qtarget(self, obs_input, agent_ids):
        """
        Target network forward.
        Returns: (None, q_target_values)
        """
        features = self.target_representation(obs_input)['state']
        q_input = torch.cat([features, agent_ids], dim=-1)
        q_values = self.target_Qhead(q_input)
        return None, q_values

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep.data)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep.data)


# ============================================================
# Replay Buffer (flattened multi-agent storage)
# ============================================================
class ReplayBuffer:
    """MARL off-policy replay buffer with flattened multi-agent storage."""
    def __init__(self, n_envs, buffer_size, batch_size, obs_dim, n_agents, n_actions, use_actions_mask=True):
        self.n_envs = n_envs
        self.n_size = buffer_size // n_envs
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.use_actions_mask = use_actions_mask
        self.ptr = 0
        self.size = 0
        self._init_buffers()

    def _init_buffers(self):
        shape = (self.n_envs, self.n_size)
        self.obs_buf = np.zeros(shape + (self.n_agents * self.obs_dim,), dtype=np.float32)
        self.act_buf = np.zeros(shape + (self.n_agents,), dtype=np.int64)
        self.obs_next_buf = np.zeros(shape + (self.n_agents * self.obs_dim,), dtype=np.float32)
        self.rew_buf = np.zeros(shape + (self.n_agents,), dtype=np.float32)
        self.ter_buf = np.zeros(shape + (self.n_agents,), dtype=np.bool_)
        self.mask_buf = np.ones(shape + (self.n_agents,), dtype=np.bool_)
        self.avail_buf = None
        self.avail_next_buf = None
        if self.use_actions_mask:
            self.avail_buf = np.ones(shape + (self.n_agents * self.n_actions,), dtype=np.bool_)
            self.avail_next_buf = np.ones(shape + (self.n_agents * self.n_actions,), dtype=np.bool_)

    def store(self, obs, actions, obs_next, rewards, terminals, agent_mask=None,
              avail_actions=None, avail_actions_next=None):
        idx = self.ptr
        self.obs_buf[:, idx] = obs
        self.act_buf[:, idx] = actions
        self.obs_next_buf[:, idx] = obs_next
        self.rew_buf[:, idx] = rewards
        self.ter_buf[:, idx] = terminals
        if agent_mask is not None:
            self.mask_buf[:, idx] = agent_mask
        if self.use_actions_mask and avail_actions is not None:
            self.avail_buf[:, idx] = avail_actions
            self.avail_next_buf[:, idx] = avail_actions_next
        self.ptr = (self.ptr + 1) % self.n_size
        self.size = min(self.size + 1, self.n_size)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        env_choices = np.random.choice(self.n_envs, batch_size)
        step_choices = np.random.choice(self.size, batch_size)
        sample = {
            'obs': self.obs_buf[env_choices, step_choices],
            'actions': self.act_buf[env_choices, step_choices],
            'obs_next': self.obs_next_buf[env_choices, step_choices],
            'rewards': self.rew_buf[env_choices, step_choices],
            'terminals': self.ter_buf[env_choices, step_choices],
            'agent_mask': self.mask_buf[env_choices, step_choices],
            'batch_size': batch_size,
        }
        if self.use_actions_mask:
            sample['avail_actions'] = self.avail_buf[env_choices, step_choices]
            sample['avail_actions_next'] = self.avail_next_buf[env_choices, step_choices]
        return sample


# ============================================================
# Parallel Environment Manager
# ============================================================
class ParallelEnv:
    """Manages multiple env instances for vectorized training."""
    def __init__(self, config):
        self.config = config
        self.num_envs = config.parallels
        self.envs = []
        for i in range(self.num_envs):
            env = gym.make(config.env_id, max_steps=config.max_episode_steps)
            self.envs.append(env)

        self.n_agents = len(self.envs[0].action_space)
        self.agents = [f'agent_{i}' for i in range(self.n_agents)]
        self.obs_dim = self.envs[0].observation_space[0].shape[0]
        self.n_actions = self.envs[0].action_space[0].n
        self.current_step = 0

    def reset(self):
        """Reset all environments. Returns obs_dict (list of dicts)."""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.config.env_seed + i)
            obs_list.append(obs)
        obs_dict = [{k: obs_list[e][i] for i, k in enumerate(self.agents)}
                    for e in range(self.num_envs)]
        return obs_dict

    def step(self, actions_dict):
        """
        Step all environments.
        Args:
            actions_dict: list of dicts [{agent_0: action, ...}, ...]
        Returns: next_obs, rewards, terminated, truncated, info
        """
        next_obs_list = []
        rew_list = []
        ter_list = []
        trunc_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            acts = [int(actions_dict[i][k]) for k in self.agents]
            obs, rew, terminated, truncated, info = env.step(acts)
            next_obs_list.append(obs)
            rew_list.append(rew)
            ter_list.append(terminated)
            trunc_list.append(truncated)
            info_list.append(info)

        # Build output dicts
        buf_next_obs = [{k: next_obs_list[e][i] for i, k in enumerate(self.agents)}
                        for e in range(self.num_envs)]
        buf_rewards = [{k: float(rew_list[e][i]) for i, k in enumerate(self.agents)}
                       for e in range(self.num_envs)]
        buf_terminated = [{k: bool(ter_list[e]) for k in self.agents}
                          for e in range(self.num_envs)]
        buf_truncated = [bool(t) for t in trunc_list]

        # Build info with episode tracking
        buf_info = []
        for i in range(self.num_envs):
            info = {
                'agent_mask': {k: True for k in self.agents},
                'episode_step': 0,
                'episode_score': {k: rew_list[i][j] for j, k in enumerate(self.agents)},
            }
            # Handle episode termination: auto-reset
            if all(buf_terminated[i].values()) or buf_truncated[i]:
                reset_obs, _ = self.envs[i].reset(
                    seed=self.config.env_seed + i + self.current_step + 1)
                info['reset_obs'] = {k: reset_obs[j] for j, k in enumerate(self.agents)}
                if self.config.use_actions_mask:
                    info['reset_avail_actions'] = {
                        k: np.ones(self.n_actions, dtype=np.bool_) for k in self.agents}
            buf_info.append(info)

        return buf_next_obs, buf_rewards, buf_terminated, buf_truncated, buf_info

    def close(self):
        for env in self.envs:
            env.close()


# ============================================================
# IQL Agent
# ============================================================
class IQLAgent:
    def __init__(self, config, envs):
        self.config = config
        self.envs = envs
        self.device = torch.device(config.device)

        self.n_agents = envs.n_agents
        self.agent_keys = envs.agents
        self.obs_dim = envs.obs_dim
        self.n_actions = envs.n_actions

        # Epsilon-greedy
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / max(1, config.decay_step_greedy)
        self.e_greedy = self.start_greedy

        self.start_training = config.start_training
        self.training_frequency = config.training_frequency
        self.current_step = 0
        self.current_episode = np.zeros(config.parallels, dtype=np.int32)
        self.episode_scores = deque(maxlen=100)
        self._episode_rewards = np.zeros(config.parallels, dtype=np.float32)

        # Build policy
        self.policy = BasicQnetwork(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            representation_hidden_size=config.representation_hidden_size,
            q_hidden_size=config.q_hidden_size,
            activation=nn.ReLU,
            device=self.device,
        )

        # Build optimizer (same as xuance: Adam with eps=1e-5)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters_model, config.learning_rate, eps=1e-5)

        # Build replay buffer
        self.memory = ReplayBuffer(
            n_envs=config.parallels,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            use_actions_mask=config.use_actions_mask,
        )

    def _obs_to_flat(self, obs_dict):
        """Convert list of dict observations to flat numpy array.
        obs_dict: [{agent_0: obs, agent_1: obs, ...}, ...] (length: n_envs)
        Returns: [n_envs, n_agents * obs_dim]
        """
        batch = len(obs_dict)
        return np.array([np.concatenate([obs_dict[e][k] for k in self.agent_keys])
                         for e in range(batch)], dtype=np.float32)

    def _acts_to_flat(self, actions_dict):
        """Convert actions dict to flat array.
        actions_dict: [{agent_0: action, ...}, ...]
        Returns: [n_envs, n_agents]
        """
        batch = len(actions_dict)
        return np.array([[actions_dict[e][k] for k in self.agent_keys]
                         for e in range(batch)], dtype=np.int64)

    def _rews_to_flat(self, rewards_dict):
        """Convert rewards dict to flat array.
        Returns: [n_envs, n_agents]
        """
        batch = len(rewards_dict)
        return np.array([[rewards_dict[e][k] for k in self.agent_keys]
                         for e in range(batch)], dtype=np.float32)

    def _ters_to_flat(self, terminated_dict):
        """Convert terminated dict to flat array.
        Returns: [n_envs, n_agents]
        """
        batch = len(terminated_dict)
        return np.array([[terminated_dict[e][k] for k in self.agent_keys]
                         for e in range(batch)], dtype=np.bool_)

    def _avail_to_flat(self, avail_dict):
        """Convert avail_actions dict to flat array.
        Returns: [n_envs, n_agents * n_actions]
        """
        batch = len(avail_dict)
        return np.array([np.concatenate(
            [avail_dict[e][k] for k in self.agent_keys])
            for e in range(batch)], dtype=np.bool_)

    def _get_actions(self, obs_dict, test_mode=False):
        """
        Get actions for all parallel envs.
        obs_dict: [{agent_0: obs, ...}, ...]
        Returns: list of dicts [{agent_0: action, ...}, ...]
        """
        batch_size = len(obs_dict)
        bs = batch_size * self.n_agents

        # Flatten observations: [batch, n_agents * obs_dim]
        obs_flat = self._obs_to_flat(obs_dict)
        obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)

        # Create agent IDs: [batch * n_agents, n_agents]
        agents_id = np.eye(self.n_agents, dtype=np.float32).reshape(1, self.n_agents, self.n_agents)
        agents_id = agents_id.repeat(batch_size, axis=0).reshape(bs, self.n_agents)
        agents_id_tensor = torch.as_tensor(agents_id, dtype=torch.float32, device=self.device)

        # Flatten obs for network: [batch * n_agents, obs_dim]
        obs_network = obs_flat.reshape(bs, self.obs_dim)
        obs_network_tensor = torch.as_tensor(obs_network, dtype=torch.float32, device=self.device)

        # Inference
        with torch.no_grad():
            _, actions_tensor, q_values_tensor = self.policy(obs_network_tensor, agents_id_tensor)

        actions_np = actions_tensor.cpu().numpy()
        actions_np = actions_np.reshape(batch_size, self.n_agents)
        actions_dict = [{k: int(actions_np[e, i]) for i, k in enumerate(self.agent_keys)}
                        for e in range(batch_size)]

        # Epsilon-greedy exploration (only during training)
        if not test_mode:
            actions_dict = self._exploration(batch_size, actions_dict)

        return actions_dict

    def _exploration(self, batch_size, actions_dict):
        """Epsilon-greedy exploration."""
        if np.random.rand() < self.e_greedy:
            for e in range(batch_size):
                for k in self.agent_keys:
                    actions_dict[e][k] = int(np.random.randint(self.n_actions))
        return actions_dict

    def _update_explore_factor(self):
        if self.e_greedy > self.end_greedy:
            self.e_greedy = max(self.end_greedy,
                                self.start_greedy - self.delta_egreedy * self.current_step)

    def _store_experience(self, obs_dict, actions_dict, next_obs_dict, rewards_dict,
                          terminated_dict, info):
        """Store a step of experience in the replay buffer."""
        obs_flat = self._obs_to_flat(obs_dict)
        act_flat = self._acts_to_flat(actions_dict)
        obs_next_flat = self._obs_to_flat(next_obs_dict)
        rew_flat = self._rews_to_flat(rewards_dict)
        ter_flat = self._ters_to_flat(terminated_dict)
        mask_flat = np.ones((len(obs_dict), self.n_agents), dtype=np.bool_)

        self.memory.store(obs_flat, act_flat, obs_next_flat, rew_flat, ter_flat, mask_flat)

    def _update_policy(self):
        """Sample from replay buffer and update policy."""
        if self.memory.size < self.memory.batch_size:
            return {}

        sample = self.memory.sample()
        bs = sample['batch_size']
        bs_expanded = bs * self.n_agents

        # Convert to tensors
        device = self.device
        obs_flat_t = torch.as_tensor(sample['obs'], dtype=torch.float32, device=device)
        act_t = torch.as_tensor(sample['actions'], dtype=torch.long, device=device)
        obs_next_flat_t = torch.as_tensor(sample['obs_next'], dtype=torch.float32, device=device)
        rew_t = torch.as_tensor(sample['rewards'], dtype=torch.float32, device=device)
        ter_t = torch.as_tensor(sample['terminals'], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(sample['agent_mask'], dtype=torch.float32, device=device)

        # Reshape to per-agent: [bs, n_agents, ...] -> [bs * n_agents, ...]
        obs_net = obs_flat_t.reshape(bs_expanded, self.obs_dim)
        obs_next_net = obs_next_flat_t.reshape(bs_expanded, self.obs_dim)
        act_flat = act_t.reshape(bs_expanded)
        rew_flat = rew_t.reshape(bs_expanded)
        ter_flat = ter_t.reshape(bs_expanded)
        mask_flat = mask_t.reshape(bs_expanded)

        # Agent IDs: [bs * n_agents, n_agents]
        agents_id = torch.eye(self.n_agents, device=device).unsqueeze(0).expand(bs, -1, -1)
        agents_id = agents_id.reshape(bs_expanded, self.n_agents)

        # ---- IQL update (same logic as xuance's IQL_Learner) ----

        # Q(s,a) from eval network
        _, _, q_eval = self.policy(obs_net, agents_id)
        q_eval_a = q_eval.gather(-1, act_flat.unsqueeze(-1)).reshape(bs_expanded)

        # Q(s',a') from target network
        _, q_next = self.policy.Qtarget(obs_next_net, agents_id)

        # Double Q-learning
        if self.config.double_q:
            with torch.no_grad():
                _, actions_next, _ = self.policy(obs_next_net, agents_id)
            q_next_a = q_next.gather(-1, actions_next.unsqueeze(-1).long()).reshape(bs_expanded)
        else:
            q_next_a = q_next.max(dim=-1, keepdim=True).values.reshape(bs_expanded)

        # TD target: r + gamma * (1-done) * Q(s',a')
        q_target = rew_flat + (1 - ter_flat) * self.config.gamma * q_next_a

        # MSE loss with mask
        td_error = (q_eval_a - q_target.detach()) * mask_flat
        loss = (td_error ** 2).sum() / mask_flat.sum().clamp(min=1)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.config.grad_clip_norm)
        self.optimizer.step()

        info = {
            'loss_Q': loss.item(),
            'predictQ': q_eval_a.mean().item(),
        }
        return info

    def train(self, train_steps):
        """Main training loop."""
        from tqdm import tqdm

        # Reset environments
        obs_dict = self.envs.reset()
        self._update_counter = 0

        pbar = tqdm(total=train_steps, desc="Training")
        steps_done = 0

        while steps_done < train_steps:
            n_steps_this_iter = self.envs.num_envs

            # Get actions
            actions_dict = self._get_actions(obs_dict, test_mode=False)

            # Step environments
            next_obs_dict, rewards_dict, terminated_dict, truncated, info = \
                self.envs.step(actions_dict)

            # Store experience
            self._store_experience(
                obs_dict, actions_dict, next_obs_dict,
                rewards_dict, terminated_dict, info)

            # Update policy
            train_info = {}
            if self.current_step >= self.start_training and \
               self.current_step % self.training_frequency == 0:
                train_info = self._update_policy()
                self._update_counter += 1
                if self._update_counter % self.config.sync_frequency == 0:
                    self.policy.copy_target()

            # Accumulate episode rewards
            for i in range(self.envs.num_envs):
                rew_mean = np.mean([rewards_dict[i][k] for k in self.agent_keys])
                self._episode_rewards[i] += rew_mean

            # Handle episode termination
            for i in range(self.envs.num_envs):
                if all(terminated_dict[i].values()) or truncated[i]:
                    self.episode_scores.append(self._episode_rewards[i])
                    self._episode_rewards[i] = 0.0
                    self.current_episode[i] += 1

                    if 'reset_obs' in info[i]:
                        next_obs_dict[i] = info[i]['reset_obs']

            obs_dict = deepcopy(next_obs_dict)

            self.current_step += n_steps_this_iter
            steps_done += 1
            self._update_explore_factor()

            # Logging
            if steps_done % max(1, train_steps // 200) == 0:
                avg_score = np.mean(self.episode_scores) if self.episode_scores else 0.0
                pbar.set_postfix({
                    'step': self.current_step,
                    'eps': f'{self.e_greedy:.3f}',
                    'score': f'{avg_score:.3f}',
                    'buf': self.memory.size,
                    'loss': f'{train_info.get("loss_Q", 0):.4f}',
                })
            pbar.update(1)

        pbar.close()
        print(f"\nTraining complete. Total steps: {self.current_step}")
        if self.episode_scores:
            print(f"Average episode score (last 100): {np.mean(self.episode_scores):.3f}")

    def evaluate(self, n_episodes=5):
        """Evaluate current policy without exploration."""
        test_envs = [gym.make(self.config.env_id, max_steps=self.config.max_episode_steps)
                     for _ in range(n_episodes)]
        total_rewards = []
        for env in test_envs:
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_dict = [{k: obs[i] for i, k in enumerate(self.agent_keys)}]
                actions_dict = self._get_actions(obs_dict, test_mode=True)
                acts = [int(actions_dict[0][k]) for k in self.agent_keys]
                obs, rew, terminated, truncated, _ = env.step(acts)
                ep_reward += float(np.mean(rew))
                done = bool(terminated or truncated)
            total_rewards.append(ep_reward)
            env.close()
        mean_rew = float(np.mean(total_rewards)) if total_rewards else 0.0
        print(f"Evaluation over {n_episodes} episodes: avg reward = {mean_rew:.3f}")
        return mean_rew

    def save_model(self, path=None):
        if path is None:
            path = self.config.model_dir
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, 'iql_model.pth')
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'n_agents': self.n_agents,
                'n_actions': self.n_actions,
            },
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, path=None):
        if path is None:
            path = self.config.model_dir
        filepath = os.path.join(path, 'iql_model.pth')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {filepath}")

    def run_inference(self, n_episodes=10, render=False):
        """Run inference with the trained model."""
        env = gym.make(self.config.env_id, max_steps=self.config.max_episode_steps)
        all_rewards = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            while not done:
                if render:
                    env.render()
                obs_dict = [{k: obs[i] for i, k in enumerate(self.agent_keys)}]
                actions_dict = self._get_actions(obs_dict, test_mode=True)
                acts = [int(actions_dict[0][k]) for k in self.agent_keys]
                obs, rew, terminated, truncated, _ = env.step(acts)
                ep_reward += float(np.mean(rew))
                step += 1
                done = bool(terminated or truncated)
            all_rewards.append(ep_reward)
            print(f"Episode {ep + 1}: reward={ep_reward:.3f}, steps={step}")
        env.close()
        mean_rew = float(np.mean(all_rewards)) if all_rewards else 0.0
        print(f"\nAverage reward: {mean_rew:.3f}")
        return all_rewards


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="IQL for rware (independent implementation)")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help="train or infer")
    parser.add_argument('--render', action='store_true', help="render during inference")
    parser.add_argument('--episodes', type=int, default=10, help="inference episodes")
    parser.add_argument('--model-path', type=str, default=None, help="model path")
    parser.add_argument('--steps', type=int, default=None, help="override training steps")
    parser.add_argument('--parallels', type=int, default=None, help="override parallel envs")
    args = parser.parse_args()

    import rware  # register gym envs
    config = IQLConfig()
    if args.steps:
        config.running_steps = args.steps
    if args.parallels:
        config.parallels = args.parallels
    if args.model_path:
        config.model_dir = args.model_path
    if torch.cuda.is_available():
        config.device = 'cuda'

    print(f"Device: {config.device}")
    print(f"env_id: {config.env_id}, parallels: {config.parallels}, steps: {config.running_steps}")

    if args.mode == 'train':
        print("\n===== IQL Training =====")
        envs = ParallelEnv(config)
        agent = IQLAgent(config, envs)
        try:
            agent.train(train_steps=config.running_steps // config.parallels)
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        agent.save_model()
        envs.close()
        agent.evaluate(n_episodes=config.test_episode)

    else:
        print("\n===== IQL Inference =====")
        envs = ParallelEnv(config)
        agent = IQLAgent(config, envs)
        agent.load_model()
        envs.close()
        agent.run_inference(n_episodes=args.episodes, render=args.render)


if __name__ == '__main__':
    main()
