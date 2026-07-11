#!/usr/bin/env python3
import os
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
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


# ============ Categorical Actor Net ============
class ActorNet(nn.Module):
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


# ============ Critic Net ============
class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=nn.ReLU, device=None):
        super().__init__()
        layers = []
        inp = input_dim
        for h in hidden_sizes:
            layers.extend(mlp_block(inp, h, activation, device))
            inp = h
        layers.extend(mlp_block(inp, 1, None, device))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ============ On-Policy Buffer with GAE ============
class MAPPOGAEBuffer:
    def __init__(self, buffer_size, n_agents, obs_dim, act_dim, state_dim, gamma, gae_lambda):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        self.obs = np.zeros((self.buffer_size, self.n_agents, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_agents), dtype=np.int64)
        self.log_probs = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.terminals = np.zeros((self.buffer_size, self.n_agents), dtype=np.bool_)
        self.values = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.state = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.adv = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.ret = np.zeros((self.buffer_size, self.n_agents), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, obs, actions, log_probs, rewards, terminals, values, state):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.terminals[self.ptr] = terminals
        self.values[self.ptr] = values
        self.state[self.ptr] = state
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_gae(self, last_value):
        last_value = np.asarray(last_value, dtype=np.float32)
        for i in range(self.size - 1, -1, -1):
            if i == self.size - 1:
                next_val = last_value
            else:
                next_val = self.values[i + 1]
            done_mask = 1.0 - self.terminals[i].astype(np.float32)
            delta = self.rewards[i] + self.gamma * next_val * done_mask - self.values[i]
            self.adv[i] = delta
        for i in range(self.size - 2, -1, -1):
            done_mask = 1.0 - self.terminals[i].astype(np.float32)
            self.adv[i] += self.gamma * self.gae_lambda * self.adv[i + 1] * done_mask
        self.ret = self.adv + self.values#self.return

    def sample(self, indexes):
        return {
            'obs': self.obs[indexes],
            'actions': self.actions[indexes],
            'log_probs': self.log_probs[indexes],
            'values': self.values[indexes],
            'advantages': self.adv[indexes],
            'returns': self.ret[indexes],
            'state': self.state[indexes],
            'batch_size': len(indexes),
        }

# ============ MAPPO Policy ============
class MAPPOPolicy(nn.Module):
    def __init__(self, obs_dim, n_agents, n_actions, state_dim,
                 actor_hidden, critic_hidden, activation=nn.ReLU, device=None):
        super().__init__()
        self.device = device
        self.n_agents = n_agents
        self.n_actions = n_actions

        # actor: individual obs + agent_id
        #(obs_dim + n_agents,)->h
        self.actor_rep = BasicMLP(
            input_shape=(obs_dim + n_agents,),
            hidden_sizes=actor_hidden,
            activation=activation,
            device=device
        )
        rep_out = self.actor_rep.output_shapes['state'][0]
        #(h,)->(n_agents,n_actions)
        self.actor = ActorNet(rep_out, n_actions, [], activation, device)

        # critic: global state
        self.critic_rep = BasicMLP(
            input_shape=(state_dim + n_agents,),
            hidden_sizes=critic_hidden,
            activation=activation,
            device=device
        )
        critic_rep_out = self.critic_rep.output_shapes['state'][0]
        self.critic = CriticNet(critic_rep_out, [], activation, device)

    def get_actor_out(self, obs, agent_ids):
        obs_concat = torch.cat([obs, agent_ids], dim=-1)
        rep = self.actor_rep(obs_concat)
        logits = self.actor(rep['state'])
        dist = Categorical(logits=logits)
        return dist

    def get_critic_value(self, state, agent_ids):
        state_concat = torch.cat([state, agent_ids], dim=-1)
        rep = self.critic_rep(state_concat)
        value = self.critic(rep['state'])
        return value

    def evaluate_actions(self, obs, agent_ids, actions):
        dist = self.get_actor_out(obs, agent_ids)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, dist

# ============ MAPPO Agent ============
class MAPPOAgent:
    def __init__(self, config):
        self.config = config

        self.device = config.device
        self.n_agents = config.n_agents
        self.obs_dim = config.obs_dim
        self.act_dim = config.act_dim
        self.state_dim = config.state_dim

        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_range = config.clip_range
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.value_clip_range = getattr(config, 'value_clip_range', self.clip_range)
        self.use_value_clip = getattr(config, 'use_value_clip', True)
        self.use_huber_loss = getattr(config, 'use_huber_loss', False)
        self.huber_delta = getattr(config, 'huber_delta', 10.0)

        self.n_epochs = config.n_epochs
        self.n_minibatch = getattr(config, 'n_minibatch', 1)
        self.batch_size = config.buffer_size // self.n_minibatch
        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm

        self.policy = MAPPOPolicy(
            obs_dim=self.obs_dim,
            n_agents=self.n_agents,
            n_actions=self.act_dim,
            state_dim=self.state_dim,
            actor_hidden=config.actor_hidden_size,
            critic_hidden=config.critic_hidden_size,
            activation=nn.ReLU,
            device=self.device
        ).to(self.device)

        self.memory = MAPPOGAEBuffer(
            buffer_size=config.buffer_size,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), config.learning_rate, eps=1e-5)
        self.use_linear_lr_decay = getattr(config, 'use_linear_lr_decay', False)
        self.end_factor_lr_decay = getattr(config, 'end_factor_lr_decay', 0.5)
        if self.use_linear_lr_decay:
            total_iters = config.running_steps // config.buffer_size
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=self.end_factor_lr_decay, total_iters=total_iters
            )
        else:
            self.scheduler = None
        self.iterations = 0

    @torch.no_grad()
    def get_actions(self, obs_np, state_np, deterministic=False):
        bs = obs_np.shape[0]
        agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

        dist = self.policy.get_actor_out(obs_t, agent_ids)
        if deterministic:
            actions = dist.probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # compute critic values
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        state_tiled = state_t.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(bs * self.n_agents, -1)
        agent_ids_flat = agent_ids.reshape(bs * self.n_agents, -1)
        values = self.policy.get_critic_value(state_tiled, agent_ids_flat)
        values = values.reshape(bs, self.n_agents)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def get_deterministic_actions(self, obs_np):
        bs = obs_np.shape[0]
        agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        dist = self.policy.get_actor_out(obs_t, agent_ids)
        actions = dist.probs.argmax(dim=-1)
        return actions.cpu().numpy()

    @torch.no_grad()
    def get_V(self, state_np):
        bs = 1
        agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
        agent_ids_flat = agent_ids.reshape(bs * self.n_agents, -1)
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        state_tiled = state_t.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(bs * self.n_agents, -1)
        values = self.policy.get_critic_value(state_tiled, agent_ids_flat)
        return values.reshape(bs, self.n_agents).cpu().numpy()

    def store_experience(self, obs, actions, log_probs, rewards, terminals, values, state):
        self.memory.store(obs, actions, log_probs, rewards, terminals, values, state)

    def train_epoch(self, batch):
        bs = batch['batch_size']

        agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(bs, -1, -1)
        agent_ids_flat = agent_ids.reshape(bs * self.n_agents, -1)

        obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        obs_flat = obs.reshape(bs * self.n_agents, -1)
        actions = torch.as_tensor(batch['actions'], dtype=torch.int64, device=self.device)
        actions_flat = actions.reshape(-1)

        old_log_probs = torch.as_tensor(batch['log_probs'], dtype=torch.float32, device=self.device)
        old_log_probs_flat = old_log_probs.reshape(-1)
        advantages = torch.as_tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        advantages_flat = advantages.reshape(-1)
        returns = torch.as_tensor(batch['returns'], dtype=torch.float32, device=self.device)
        returns_flat = returns.reshape(-1)
        state = torch.as_tensor(batch['state'], dtype=torch.float32, device=self.device)
        state_tiled = state.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(bs * self.n_agents, -1)
        old_values = torch.as_tensor(batch['values'], dtype=torch.float32, device=self.device)
        old_values_flat = old_values.reshape(-1)

        # normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # actor loss
        log_probs_flat, entropy_flat, _ = self.policy.evaluate_actions(obs_flat, agent_ids_flat, actions_flat)
        ratio = torch.exp(log_probs_flat - old_log_probs_flat)
        loss_a1 = ratio * advantages_flat
        loss_a2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_flat
        loss_a = -torch.min(loss_a1, loss_a2).mean()

        # entropy loss
        loss_e = entropy_flat.mean()

        # critic loss
        values_flat = self.policy.get_critic_value(state_tiled, agent_ids_flat).reshape(-1)
        if self.use_value_clip:
            value_clipped = old_values_flat + (values_flat - old_values_flat).clamp(
                -self.value_clip_range, self.value_clip_range)
            if self.use_huber_loss:
                huber = nn.HuberLoss(reduction='none', delta=self.huber_delta)
                loss_v1 = huber(values_flat, returns_flat)
                loss_v2 = huber(value_clipped, returns_flat)
            else:
                loss_v1 = (values_flat - returns_flat) ** 2
                loss_v2 = (value_clipped - returns_flat) ** 2
            loss_c = torch.max(loss_v1, loss_v2).mean()
        else:
            if self.use_huber_loss:
                huber = nn.HuberLoss(reduction='mean', delta=self.huber_delta)
                loss_c = huber(values_flat, returns_flat)
            else:
                loss_c = ((values_flat - returns_flat) ** 2).mean()

        loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e

        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.iterations += 1
        return loss_a.item(), loss_c.item(), loss_e.item()

    def save_model(self, path):
        #os.makedirs(os.path.dirname(path), exist_ok=True)
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

    #自定义奖励函数: 对原始奖励进行塑形，鼓励高效完成任务
    def reward_fn(self, reward):
        from rware.warehouse import Action
        """
        class Action(Enum):
            NOOP = 0
            FORWARD = 1
            LEFT = 2
            RIGHT = 3
            TOGGLE_LOAD = 4  # 装载/放下货架
        """
        unwrapped_env=self.env.unwrapped
        reward = np.array(reward, dtype=np.float32)
        # 1. 惩罚NOOP（原地不动），鼓励持续移动
        for agent in unwrapped_env.agents:
            if agent.req_action == Action.NOOP:
                reward[agent.id - 1] -= 0.025
        # 2. 搬运请求队列中的货架时给予微小奖励，引导智能体关注有效目标
        for agent in unwrapped_env.agents:
            if agent.carrying_shelf is not None:
                if agent.carrying_shelf in unwrapped_env.request_queue:
                    reward[agent.id - 1] += 0.015
                else:
                    reward[agent.id - 1] -= 0.005
        # 3. 每步微小时间惩罚，鼓励尽快完成配送
        reward -= 0.02
        return reward

def get_global_state(obs):
    return obs.reshape(-1)


def draw(ep_total_rewards, ep_avg_rewards, ep_losses, ep_steps, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 14))
    axes[0].plot(ep_total_rewards, color='tab:blue')
    axes[0].set_ylabel('Episode Total Reward')
    axes[0].set_xlabel('Episode')
    axes[0].grid(True)

    axes[1].plot(ep_avg_rewards, color='tab:orange')
    axes[1].set_ylabel('Average Reward (per step)')
    axes[1].set_xlabel('Episode')
    axes[1].grid(True)

    axes[2].plot(ep_steps, color='tab:purple')
    axes[2].set_ylabel('Episode Steps')
    axes[2].set_xlabel('Episode')
    axes[2].grid(True)

    axes[3].plot(ep_losses, color='tab:green')
    axes[3].set_ylabel('Average Loss')
    axes[3].set_xlabel('Update Step')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def train(config):
    env = RoboticWarehouseEnv(config)
    agent = MAPPOAgent(config)
    n_agents = config.n_agents

    obs = env.reset()
    state = get_global_state(obs)

    episode_total_reward = 0.0
    episode_steps = 0
    total_steps = 0
    train_episode = 0
    update_count = 0

    pbar = range(1, config.running_steps + 1)
    if config.use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(pbar)

    best_avg_reward = -1e9
    ep_total_rewards = []
    ep_avg_rewards = []
    ep_losses = []
    ep_step_counts = []
    current_losses = []

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config.model_dir}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    for step in pbar:
        obs_batch = obs[np.newaxis, :]  # (1, n_agents, obs_dim)
        state_batch = state[np.newaxis, :]  # (1, state_dim)
        actions, log_probs, values = agent.get_actions(obs_batch, state_batch)
        o, r, d, tr, _ = env.step(actions[0])
        done = d or tr

        terminal = np.full(n_agents, done)
        obs_next = o
        state_next = get_global_state(obs_next)

        agent.store_experience(obs, actions[0], log_probs[0], r, terminal, values[0], state)

        # 对奖励进行自定义处理
        r = env.reward_fn(r)
        episode_total_reward += r.sum()
        episode_steps += 1

        obs = obs_next
        state = state_next
        total_steps += 1

        if done or episode_steps >= config.max_episode_steps:
            if done:
                last_value = np.zeros(n_agents, dtype=np.float32)
            else:
                last_val = agent.get_V(state[np.newaxis, :])
                last_value = last_val.ravel()
            agent.memory.compute_gae(last_value)

            #minibatch训练 例如 [0,64,128,255,384,511,640...]
            mem_size = agent.memory.size
            indexes = np.arange(mem_size)
            for _ in range(config.n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, mem_size, agent.batch_size):
                    end = min(start + agent.batch_size, mem_size)
                    if end - start < 2:
                        continue
                    batch_idx = indexes[start:end]
                    batch = agent.memory.sample(batch_idx)
                    loss_a, loss_c, loss_e = agent.train_epoch(batch)
                    current_losses.append(loss_a + loss_c - loss_e)
                    update_count += 1

            train_episode += 1
            avg_r = episode_total_reward / max(1, episode_steps)
            avg_loss = np.mean(current_losses) if current_losses else 0.0

            ep_total_rewards.append(episode_total_reward)
            ep_avg_rewards.append(avg_r)
            ep_losses.append(avg_loss)
            ep_step_counts.append(episode_steps)

            if avg_r > best_avg_reward:
                best_avg_reward = avg_r

            if config.use_tqdm:
                pbar.set_description(
                    f"Ep {train_episode} R={episode_total_reward/n_agents:.4f} avgR={avg_r:.4f} "
                    f"Loss={avg_loss:.4f} Steps={episode_steps}"
                )

            agent.memory.clear()
            episode_total_reward = 0.0
            episode_steps = 0
            current_losses = []
            obs = env.reset()
            state = get_global_state(obs)

        if step % config.eval_interval == 0 and step > 0 and train_episode > 0:
            eval_reward = evaluate(config, agent)
            print(f"Evaluation at step {step}: avg_reward = {eval_reward:.4f}")
            agent.save_model(f"{save_dir}/{best_avg_reward:.4f}_step_{step}.pth")

    env.close()

    agent.save_model(f"{save_dir}/{best_avg_reward:.4f}_final.pth")
    print(f"Training completed. Model saved to {save_dir}")
    draw(ep_total_rewards, ep_avg_rewards, ep_losses, ep_step_counts, save_dir)


@torch.no_grad()
def evaluate(config, agent=None,is_pause=False):
    if agent is None:
        agent = MAPPOAgent(config)
        try:
            agent.load_model(f"{config.load_dir}")
            print(f"Load model from {config.load_dir} success")
        except:
            print(f"Load model from {config.load_dir} failed")
    #print(f"Evaluation mode: {config.render_mode}")
    env = RoboticWarehouseEnv(config)
    ep_rewards = []
    for ep in range(config.test_episode):
        obs = env.reset()
        env.env.render()
        done = False
        ep_r = 0
        steps = 0
        while not done and steps < config.max_episode_steps:
            obs_t = obs[np.newaxis, :]
            actions = agent.get_deterministic_actions(obs_t)
            actions = actions[0]
            #print(f"Actions: {actions}")
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            ep_r += reward.sum()
            steps += 1
            env.env.render()
            if is_pause:
                os.system("pause")
        ep_rewards.append(ep_r)
        print(f"Test Episode {ep}: reward = {ep_r:.4f}, steps = {steps}")
        if is_pause:
            os.system("pause")
    env.close()
    return np.mean(ep_rewards)


def main():
    config_path = os.path.join(os.path.dirname(__file__), 'mappo-rware-tiny-4ag-v2.yaml')

    algo_config = load_yaml(config_path)
    config = SN(**algo_config)

    config.device = getattr(config, 'device', 'cpu')
    if config.device == 'cpu':
        config.device = 'cpu'
    elif 'cuda' in config.device:
        config.device = config.device if torch.cuda.is_available() else 'cpu'

    temp_env = gym.make(config.env_id)
    config.n_agents = len(temp_env.action_space)
    config.obs_dim = temp_env.observation_space[0].shape[0]
    config.act_dim = temp_env.action_space[0].n
    config.state_dim = config.obs_dim * config.n_agents
    config.use_tqdm = True
    temp_env.close()

    print(f"MAPPO on {config.env_id}")
    print(f"  Agents: {config.n_agents}, ObsDim: {config.obs_dim}, ActDim: {config.act_dim}, StateDim: {config.state_dim}")
    print(f"  Device: {config.device}")
    print(f"  Running steps: {config.running_steps}")
    bs = config.buffer_size // getattr(config, 'n_minibatch', 1)
    print(f"  Buffer size: {config.buffer_size}, Minibatch: {getattr(config, 'n_minibatch', 1)} (batch_size={bs})")
    print(f"  PPO epochs: {config.n_epochs}, Clip range: {config.clip_range}")

    mode = getattr(config, 'mode', 'train')
    if mode == 'train':
        train(config)
    else:
        eval_reward = evaluate(config,is_pause=True)
        print(f"Evaluation: avg_reward = {eval_reward:.4f}")


if __name__ == '__main__':
    main()
