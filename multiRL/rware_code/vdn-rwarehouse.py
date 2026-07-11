import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import rware
from collections import deque
from copy import deepcopy
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'#允许加载多个 OpenMP 运行时库实例

class ReplayBuffer:
    def __init__(self, capacity, n_agents):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, obs_next, terminated, agent_ids):
        self.buffer.append((obs, actions, rewards, obs_next, terminated, agent_ids))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs_b, act_b, rew_b, obs_next_b, term_b, ids_b = [], [], [], [], [], []
        for i in indices:
            o, a, r, o_n, t, ids = self.buffer[i]
            obs_b.append(o)
            act_b.append(a)
            rew_b.append(r)
            obs_next_b.append(o_n)
            term_b.append(t)
            ids_b.append(ids)
        return (np.stack(obs_b), np.stack(act_b), np.stack(rew_b),
                np.stack(obs_next_b), np.stack(term_b), np.stack(ids_b))

    def __len__(self):
        return len(self.buffer)

def mlp_block(input_dim, output_dim, activation=None, initialize=None, device=None):
    block = []
    linear = nn.Linear(input_dim, output_dim, device=device)
    if initialize is not None:
        initialize(linear.weight)
        nn.init.constant_(linear.bias, 0)
    block.append(linear)
    if activation is not None:
        block.append(activation())
    return block, (output_dim,)

class BasicMLP(nn.Module):
    def __init__(self, input_shape, hidden_sizes, activation=None, device=None):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.device = device
        self.output_shapes = {'state': (hidden_sizes[-1],)}
        self.model = self._create_network()

    def _create_network(self):
        layers = []
        input_shape = self.input_shape
        for h in self.hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, self.activation, device=self.device)
            layers.extend(mlp)
        return nn.Sequential(*layers)

    def forward(self, observations):
        tensor_observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return {'state': self.model(tensor_observation)}

class BasicQhead(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_sizes, activation=None, device=None):
        super().__init__()
        layers_ = []
        input_shape = (state_dim,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, activation, device=device)
            layers_.extend(mlp)
        layers_.extend(mlp_block(input_shape[0], n_actions, None, device=device)[0])
        self.model = nn.Sequential(*layers_)

    def forward(self, x):
        return self.model(x)

class VDNMixer(nn.Module):
    #简单相加
    def forward(self, values_n, states=None):
        return values_n.sum(dim=1)

class VDN_policy(nn.Module):
    def __init__(self, action_space, n_agents, representation, mixer,
                 hidden_size, activation=None, device=None, use_parameter_sharing=True):
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.n_agents = n_agents
        self.use_parameter_sharing = use_parameter_sharing
        if use_parameter_sharing:
            self.model_keys = ['agent']
        else:
            self.model_keys = list(action_space.keys())

        #特征提取 representation
        self.representation = nn.ModuleDict()
        self.target_representation = nn.ModuleDict()
        for key in self.model_keys:
            self.representation[key] = representation
            self.target_representation[key] = deepcopy(representation)

        self.representation_info_shape = {key: representation.output_shapes for key in self.model_keys}

        self.eval_Qhead = nn.ModuleDict()
        self.target_Qhead = nn.ModuleDict()
        #每个model都配一对Qhead和target_Qhead
        for key in self.model_keys:
            dim_input_Q = self.representation_info_shape[key]['state'][0]
            if use_parameter_sharing:
                dim_input_Q += n_agents
            n_actions_val = list(action_space.values())[0].n
            self.eval_Qhead[key] = BasicQhead(dim_input_Q, n_actions_val, hidden_size, activation, device)
            self.target_Qhead[key] = deepcopy(self.eval_Qhead[key])

        #VDNMixer
        self.eval_Qtot = mixer
        self.target_Qtot = deepcopy(self.eval_Qtot)

    @property
    def parameters_model(self):
        params = list(self.eval_Qtot.parameters())
        for key in self.model_keys:
            params += list(self.representation[key].parameters())
            params += list(self.eval_Qhead[key].parameters())
        return params

    def forward(self, observation, agent_ids=None, avail_actions=None, agent_key=None):
        rnn_hidden_new, argmax_action, evalQ = {}, {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]

        for key in agent_list:
            outputs = self.representation[key](observation[key])
            rnn_hidden_new[key] = None

            if self.use_parameter_sharing:
                q_inputs = torch.cat([outputs['state'], agent_ids], dim=-1)
            else:
                q_inputs = outputs['state']

            evalQ[key] = self.eval_Qhead[key](q_inputs)

            if avail_actions is not None:
                evalQ_detach = evalQ[key].clone().detach()
                evalQ_detach[avail_actions[key] == 0] = -1e10
                argmax_action[key] = evalQ_detach.argmax(dim=-1, keepdim=False)
            else:
                argmax_action[key] = evalQ[key].argmax(dim=-1, keepdim=False)

        return rnn_hidden_new, argmax_action, evalQ

    def Qtarget(self, observation, agent_ids=None, agent_key=None):
        rnn_hidden_new, q_target = {}, {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs = self.target_representation[key](observation[key])
            rnn_hidden_new[key] = None
            if self.use_parameter_sharing:
                q_inputs = torch.cat([outputs['state'], agent_ids], dim=-1)
            else:
                q_inputs = outputs['state']
            q_target[key] = self.target_Qhead[key](q_inputs)
        return rnn_hidden_new, q_target

    def Q_tot(self, individual_values, states=None):
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            individual_inputs = individual_values[key].reshape([-1, self.n_agents, 1])
        else:
            individual_inputs = torch.cat([individual_values[k] for k in self.model_keys],
                                          dim=-1).reshape([-1, self.n_agents, 1])
        return self.eval_Qtot(individual_inputs, states)

    def Qtarget_tot(self, individual_values, states=None):
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            individual_inputs = individual_values[key].reshape([-1, self.n_agents, 1])
        else:
            individual_inputs = torch.cat([individual_values[k] for k in self.model_keys],
                                          dim=-1).reshape([-1, self.n_agents, 1])
        return self.target_Qtot(individual_inputs, states)

    def copy_target(self):
        for key in self.model_keys:
            for ep, tp in zip(self.representation[key].parameters(),
                              self.target_representation[key].parameters()):
                tp.data.copy_(ep.data)
            for ep, tp in zip(self.eval_Qhead[key].parameters(),
                              self.target_Qhead[key].parameters()):
                tp.data.copy_(ep.data)
        for ep, tp in zip(self.eval_Qtot.parameters(), self.target_Qtot.parameters()):
            tp.data.copy_(ep.data)

class VDNAgent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        print("using device:", self.device)
        self.n_agents = config.n_agents
        self.agent_keys = [f'agent_{i}' for i in range(self.n_agents)]
        self.model_keys = ['agent']
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.sync_frequency = config.sync_frequency
        self.double_q = config.double_q
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.decay_step_greedy = config.decay_step_greedy
        self.e_greedy = self.start_greedy
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (self.decay_step_greedy)
        self.use_grad_clip = config.use_grad_clip
        self.grad_clip_norm = config.grad_clip_norm
        self.use_actions_mask = config.use_actions_mask
        self.training_steps = 0

        obs_dim = config.obs_dim
        n_actions = config.n_actions
        action_space = {k: gym.spaces.Discrete(n_actions) for k in self.agent_keys}
        activation = nn.ReLU

        representation = BasicMLP(
            input_shape=(obs_dim,),
            hidden_sizes=config.representation_hidden_size,
            activation=activation,
            device=self.device
        )
        mixer = VDNMixer()

        self.policy = VDN_policy(
            action_space=action_space,
            n_agents=self.n_agents,
            representation=representation,
            mixer=mixer,
            hidden_size=config.q_hidden_size,
            activation=activation,
            device=self.device,
            use_parameter_sharing=True
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters_model, self.learning_rate, eps=1e-5)
        self.mse_loss = nn.MSELoss()
        self.memory = ReplayBuffer(config.buffer_size, self.n_agents)

    def get_agent_ids(self, batch_size):
        agent_ids = torch.eye(self.n_agents, device=self.device)
        agent_ids = agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return agent_ids

    def act(self, obs_tuple, avail_actions=None, eval_mode=False):
        batch_size = 1
        obs_dict = {'agent': torch.as_tensor(np.stack(obs_tuple), dtype=torch.float32, device=self.device).unsqueeze(0)}
        agent_ids = self.get_agent_ids(batch_size)

        with torch.no_grad():
            if not eval_mode and np.random.rand() < self.e_greedy:
                actions = {k: np.random.randint(0, self.policy.action_space[k].n) for k in self.agent_keys}
                return actions

            #评测或者非epislon贪心
            _, actions_tensor, _ = self.policy(obs_dict, agent_ids)
            actions = {}
            for i, k in enumerate(self.agent_keys):
                actions[k] = actions_tensor['agent'][0, i].item()
        return actions

    def train(self):
        #仅仅负责调度policy 计算Loss
        if len(self.memory) < self.config.start_training:
            return {}

        batch = self.memory.sample(self.batch_size)
        obs_np, actions_np, rewards_np, obs_next_np, terminated_np, ids_np = batch

        batch_size = self.batch_size
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        obs_next_tensor = torch.as_tensor(obs_next_np, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        terminals_tensor = torch.as_tensor(terminated_np, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)
        agent_ids = torch.as_tensor(ids_np, dtype=torch.float32, device=self.device)

        obs_dict = {'agent': obs_tensor}
        obs_next_dict = {'agent': obs_next_tensor}
        avail_actions = None

        #计算
        _, _, Q = self.policy(obs_dict, agent_ids)
        _, q_next = self.policy.Qtarget(obs_next_dict, agent_ids)

        key = 'agent'
        Q_a = Q[key].gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        if self.double_q:
            _, act_next, _ = self.policy(obs_next_dict, agent_ids)
            q_next_a = q_next[key].gather(-1, act_next[key].long().unsqueeze(-1)).squeeze(-1)
        else:
            q_next_a = q_next[key].max(dim=-1, keepdim=False).values

        rewards_mean = rewards_tensor.mean(dim=1, keepdim=True)
        terminals_tot = terminals_tensor.all(dim=1, keepdim=True).float()

        Q_tot = self.policy.Q_tot({key: Q_a})
        q_tot_next = self.policy.Qtarget_tot({key: q_next_a})
        Q_tot_target = rewards_mean + (1 - terminals_tot) * self.gamma * q_tot_next

        loss = self.mse_loss(Q_tot, Q_tot_target.detach())
        #更新
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_model, self.grad_clip_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.sync_frequency == 0:
            self.policy.copy_target()

        if self.e_greedy > self.end_greedy:
            self.e_greedy = max(self.end_greedy, self.e_greedy - self.delta_egreedy)

        return {'loss': loss.item(), 'q_tot': Q_tot.mean().item()}

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


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

    #自定义奖励函数: 对原始奖励进行塑形，鼓励高效完成任务
def reward_fn(reward,env):
    from rware.warehouse import Action
    """
    class Action(Enum):
        NOOP = 0
        FORWARD = 1
        LEFT = 2
        RIGHT = 3
        TOGGLE_LOAD = 4  # 装载/放下货架
    """

    reward = np.array(reward, dtype=np.float32)
    # 1. 惩罚NOOP（原地不动），鼓励持续移动
    for agent in env.agents:
        if agent.req_action == Action.NOOP:
            reward[agent.id - 1] -= 0.025
    # 2. 搬运请求队列中的货架时给予微小奖励，引导智能体关注有效目标
    for agent in env.agents:
        if agent.carrying_shelf is not None:
            if agent.carrying_shelf in env.request_queue:
                reward[agent.id - 1] += 0.015
            else:
                reward[agent.id - 1] -= 0.005
    # 3. 每步微小时间惩罚，鼓励尽快完成配送
    reward -= 0.02
    return reward
def train(config):
    env = gym.make(config.env_id)

    obs_sample, _ = env.reset()
    config.obs_dim = obs_sample[0].shape[0]
    config.n_actions = env.action_space[0].n
    n_agents = len(env.action_space)
    config.n_agents = n_agents

    agent = VDNAgent(config)
    agent_keys = [f'agent_{i}' for i in range(n_agents)]

    episode_total_reward = 0.0
    episode_steps = 0
    total_steps = 0
    train_episode = 0

    print("Starting VDN training on", config.env_id)
    print(f"  obs_dim={config.obs_dim}, n_agents={n_agents}, n_actions={config.n_actions}")
    print(f"  total_steps={config.running_steps}, start_training={config.start_training}")

    best_avg_reward = -1e9
    ep_total_rewards = []
    ep_avg_rewards = []
    ep_losses = []
    ep_step_counts = []
    current_losses = []

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"{config.model_dir}/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    obs_tuple, _ = env.reset()

    pbar = range(1, config.running_steps + 1)
    if config.use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(pbar)

    for step in pbar:
        actions = agent.act(obs_tuple)
        actions_list = [actions[k] for k in agent_keys]
        obs_next_tuple, r, terminated, truncated, info = env.step(actions_list)

        rewards_tuple=reward_fn(r,env.unwrapped)
        obs_np = np.stack(obs_tuple)
        obs_next_np = np.stack(obs_next_tuple)
        actions_np = np.array([actions[k] for k in agent_keys])
        rewards_np = np.array(rewards_tuple)
        terminated_np = np.array([terminated for _ in agent_keys])
        agent_ids_np = np.eye(n_agents)

        agent.memory.push(obs_np, actions_np, rewards_np, obs_next_np, terminated_np, agent_ids_np)
        total_steps += 1
        episode_total_reward += sum(rewards_tuple)
        episode_steps += 1

        train_info = agent.train()
        if train_info:
            current_losses.append(train_info['loss'])

        obs_tuple = obs_next_tuple

        if terminated or truncated:
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
                    f"Ep {train_episode} avgR={episode_total_reward/n_agents:.4f} total_avgR={avg_r:.4f} "
                    f"Loss={avg_loss:.4f} Steps={episode_steps}"
                )

            episode_total_reward = 0.0
            episode_steps = 0
            current_losses = []
            obs_tuple, _ = env.reset()

        if step % config.eval_interval == 0 and step > 0 and train_episode > 0:
            eval_reward = evaluate(config, agent)
            print(f"Evaluation at step {step}: avg_reward = {eval_reward:.4f}")
            agent.save_model(f"{save_dir}/{best_avg_reward:.4f}_step_{step}.pth")

    env.close()

    agent.save_model(f"{save_dir}/{best_avg_reward:.4f}_final.pth")
    print(f"Training completed. Model saved to {save_dir}")
    draw(ep_total_rewards, ep_avg_rewards, ep_losses, ep_step_counts, save_dir)

    final_avg = np.mean(ep_total_rewards[-100:]) if len(ep_total_rewards) >= 100 else np.mean(ep_total_rewards)
    print(f"Final average reward (last 100 episodes): {final_avg:.2f}")
    return ep_total_rewards


@torch.no_grad()
def evaluate(config, agent=None,is_pause=False):
    env = gym.make(config.env_id, render_mode=config.render_mode)

    obs_sample, _ = env.reset()
    config.obs_dim = obs_sample[0].shape[0]
    config.n_actions = env.action_space[0].n

    if agent is None:
        agent = VDNAgent(config)
        try:
            agent.load_model(config.load_dir)
            print(f"Load model from {config.load_dir} success")
        except:
            print(f"Load model from {config.load_dir} failed")

    ep_rewards = []
    for ep in range(config.test_episode):
        obs_tuple, _ = env.reset()
        done = False
        ep_r = 0
        steps = 0
        while not done and steps < config.max_episode_steps:
            actions = agent.act(obs_tuple, eval_mode=True)
            print("Actions:", actions)
            obs_next_tuple, rewards_tuple, terminated, truncated, info = env.step([actions[f'agent_{i}'] for i in range(config.n_agents)])
            done = terminated or truncated
            r=reward_fn(rewards_tuple,env.unwrapped)
            print("Rewards:", r)
            ep_r += np.mean(r)
            steps += 1
            obs_tuple = obs_next_tuple
            env.render()
            if is_pause:
                os.system("pause")
        ep_rewards.append(ep_r)
        print(f"Test Episode {ep}: reward = {ep_r:.2f}, steps = {steps}")
        if is_pause:
            os.system("pause")
    env.close()
    return np.mean(ep_rewards)

config = type('Config', (), {
    'env_id': 'rware-tiny-4ag-v2',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_grad_clip': True,
    'grad_clip_norm': 10.0,
    'use_actions_mask': False,
    'representation_hidden_size': [128, 128],
    'q_hidden_size': [128],
    'double_q': True,
    'use_tqdm': True,
    'eval_interval': 30000,
    'test_episode': 3,
    'start_greedy': 1.0,
    'end_greedy': 0.1,
    'decay_step_greedy': 200000,

    'batch_size': 256,
    'buffer_size': 20000,
    'start_training': 1000,

    'gamma': 0.99,
    'learning_rate': 0.001,
    'sync_frequency': 20,
    'running_steps': 100000,
    'max_episode_steps': 200,

    'model_dir': 'models/vdn',
    'render_mode': 'human',
    'load_dir': 'E:/RL/baseRL/multiRL/rware_code/models/vdn/20260711-124324/-0.0579_final.pth'
})()

if __name__ == "__main__":
    train(config)
    #evaluate(config, is_pause=True)
