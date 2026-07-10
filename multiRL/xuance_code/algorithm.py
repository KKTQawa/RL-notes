"""
Self-contained MARL algorithms from xuance, for study purposes.
No dependencies on xuance library — only torch, numpy, gymnasium, pettingzoo.
"""

import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import gymnasium.spaces as spaces

import pettingzoo
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.env import ParallelEnv


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def mlp_block(hidden_sizes: List[int], activation=nn.ReLU, normalize=False,
              dropout: Optional[float] = None, device=None) -> nn.Module:
    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], device=device))
        if normalize:
            layers.append(nn.LayerNorm(hidden_sizes[i + 1], device=device))
        layers.append(activation())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def space2shape(space):
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.MultiDiscrete):
        return int(space.nvec.sum())
    elif isinstance(space, spaces.Box):
        return int(space.shape[0])
    else:
        raise TypeError(f"Unknown space type: {type(space)}")


def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def _check_or_build(device, model, name):
    if model is None:
        return None
    return model


class CArgs:
    """Simple namespace for config attributes."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, item):
        return None


# ──────────────────────────────────────────────
# Representations
# ──────────────────────────────────────────────

class Basic_Identical(nn.Module):
    def __init__(self, input_shape: int, device=None):
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = input_shape

    def forward(self, x):
        return x


class Basic_MLP(nn.Module):
    def __init__(self, input_shape: int, hidden_sizes: List[int],
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        self.input_shape = input_shape
        layers = [nn.Flatten()]
        layers.append(mlp_block([input_shape] + hidden_sizes,
                                activation=activation, normalize=normalize,
                                device=device))
        self.model = nn.Sequential(*layers)
        self.output_dim = hidden_sizes[-1]

    def forward(self, x):
        return self.model(x)


def get_representation(input_shape, config, device=None):
    rep_type = config.representation if hasattr(config, 'representation') else 'Basic_MLP'
    if rep_type == 'Basic_MLP':
        hidden = config.representation_hidden if hasattr(config, 'representation_hidden') else [64, 64]
        return Basic_MLP(input_shape, hidden, device=device)
    else:
        return Basic_Identical(input_shape, device=device)


# ──────────────────────────────────────────────
# Network Heads
# ──────────────────────────────────────────────

class BasicQhead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        layers = [nn.Linear(input_dim + output_dim if output_dim is None else input_dim,
                            hidden[0], device=device)]
        # More typical: input_dim -> hidden -> output_dim
        self.model = mlp_block([input_dim] + hidden + [output_dim],
                               activation=activation, normalize=normalize,
                               device=device)
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


class CategoricalActorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        self.model = mlp_block([input_dim] + hidden + [output_dim],
                               activation=activation, normalize=normalize,
                               device=device)
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x)


class BernoulliActorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        self.model = mlp_block([input_dim] + hidden + [output_dim],
                               activation=activation, normalize=normalize,
                               device=device)
        self.output_dim = output_dim

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class GaussianActorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        self.mu = mlp_block([input_dim] + hidden + [output_dim],
                            activation=activation, normalize=normalize,
                            device=device)
        self.log_std = nn.Parameter(torch.zeros(output_dim, device=device))
        self.output_dim = output_dim

    def forward(self, x):
        mu = self.mu(x)
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mu, std


class CriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        self.model = mlp_block([input_dim] + hidden + [1],
                               activation=activation, normalize=normalize,
                               device=device)

    def forward(self, x):
        return self.model(x)


class ActorNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        self.model = mlp_block([input_dim] + hidden + [output_dim],
                               activation=activation, normalize=normalize,
                               device=device)

    def forward(self, x):
        return self.model(x)


class CriticNet_RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int] = None,
                 activation=nn.ReLU, normalize=False, device=None):
        super().__init__()
        hidden = hidden_sizes or [64, 64]
        self.rnn = nn.GRUCell(input_dim, hidden[0])
        self.fc = mlp_block([hidden[0]] + hidden[1:] + [1],
                            activation=activation, normalize=normalize,
                            device=device)

    def forward(self, x, hidden=None):
        batch = x.shape[0]
        if hidden is None:
            hidden = torch.zeros(batch, self.rnn.hidden_size, device=x.device)
        h = self.rnn(x, hidden)
        return self.fc(h), h


# ──────────────────────────────────────────────
# Mixing Networks
# ──────────────────────────────────────────────

class VDN_mixer(nn.Module):
    def __init__(self, device=None):
        super().__init__()

    def forward(self, values, states=None):
        return values.sum(dim=-1, keepdim=True)


class QMIX_mixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_hidden: int = 32,
                 hyper_hidden: int = 64, device=None):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden, device=device),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden, device=device),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden, device=device),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden * 1, device=device),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden, device=device)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden, device=device),
            nn.ReLU(),
            nn.Linear(mixing_hidden, 1, device=device),
        )

    def forward(self, values, states):
        states = states.reshape(-1, self.state_dim)
        values = values.reshape(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(states)).reshape(-1, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(states).reshape(-1, 1, self.mixing_hidden)
        hidden = torch.relu(torch.bmm(values, w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).reshape(-1, self.mixing_hidden, 1)
        b2 = self.hyper_b2(states).reshape(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).squeeze(-1)

    @property
    def mixing_hidden(self):
        return 32  # default


class QMIX_FF_mixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, mixing_hidden: int = 32,
                 device=None):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_agents, mixing_hidden, device=device),
            nn.ReLU(),
            nn.Linear(mixing_hidden, mixing_hidden, device=device),
            nn.ReLU(),
            nn.Linear(mixing_hidden, 1, device=device),
        )

    def forward(self, values, states):
        batch = values.shape[0]
        x = torch.cat([states, values], dim=-1)
        return self.net(x)


class QTRAN_mixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 hidden: int = 64, device=None):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.joint_q = nn.Sequential(
            nn.Linear(state_dim + n_agents * action_dim, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, 1, device=device),
        )
        self.joint_v = nn.Sequential(
            nn.Linear(state_dim, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, hidden, device=device),
            nn.ReLU(),
            nn.Linear(hidden, 1, device=device),
        )

    def forward(self, values, states):
        return self.joint_q(torch.cat([states, values], dim=-1))


class WQMIX_mixer(nn.Module):
    """Weighted QMIX: uses a separate 'optimal' mixing network and a weight network."""
    def __init__(self, n_agents: int, state_dim: int, mixing_hidden: int = 32,
                 hyper_hidden: int = 64, device=None):
        super().__init__()
        # Same structure as QMIX_mixer but with an additional weight network
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.qmix = QMIX_mixer(n_agents, state_dim, mixing_hidden, hyper_hidden, device)
        self.qmix_opt = QMIX_mixer(n_agents, state_dim, mixing_hidden, hyper_hidden, device)

        # Weight network: takes (state, Q_tot, Q_tot_opt) and outputs weight alpha
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim + 2, hyper_hidden, device=device),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1, device=device),
        )

    def forward(self, values, states, actions=None):
        q_tot = self.qmix(values, states)
        q_tot_opt = self.qmix_opt(values, states)
        return q_tot, q_tot_opt, self.weight_net


# ──────────────────────────────────────────────
# Policies
# ──────────────────────────────────────────────

class BasicQnetwork_marl(nn.Module):
    """Basic Q-network for IQL."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, device=None):
        super().__init__()
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        self.representation = get_representation(
            space2shape(observation_space), config, device)
        rep_dim = self.representation.output_dim
        self.action_dim = space2shape(action_space)
        self.Q = BasicQhead(rep_dim, self.action_dim,
                            hidden_sizes=config.q_hidden_sizes if hasattr(config, 'q_hidden_sizes') else [64, 64],
                            device=device)

    def forward(self, x):
        if isinstance(x, dict):
            return {k: self.Q(self.representation(v)) for k, v in x.items()}
        return self.Q(self.representation(x))

    def Qvalues(self, x):
        return self(x)

    def sample_actions(self, logits, epsilon=0.0):
        if isinstance(logits, dict):
            actions = {}
            for k, v in logits.items():
                v_np = v.cpu().numpy()
                if epsilon > 0 and np.random.random() < epsilon:
                    actions[k] = np.random.randint(0, self.action_dim, size=v.shape[0])
                else:
                    actions[k] = v_np.argmax(axis=-1)
            return actions
        if epsilon > 0 and np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim, size=logits.shape[0])
        return logits.argmax(dim=-1).cpu().numpy()


class MixingQnetwork(nn.Module):
    """Q-network with a mixing network for VDN/QMIX."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str], device=None):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = agent_keys
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

        self.agents = nn.ModuleDict()
        for key in agent_keys:
            rep = get_representation(space2shape(observation_space), config, device)
            rep_dim = rep.output_dim
            action_dim = space2shape(action_space)
            self.agents[key] = nn.Sequential(
                rep,
                BasicQhead(rep_dim, action_dim,
                           hidden_sizes=config.q_hidden_sizes if hasattr(config, 'q_hidden_sizes') else [64, 64],
                           device=device),
            )

        # Mixing network
        state_dim = config.state_space if isinstance(config.state_space, int) else space2shape(config.state_space)
        if config.mixer == 'vdn':
            self.mixer = VDN_mixer(device=device)
        elif config.mixer == 'qmix':
            self.mixer = QMIX_mixer(n_agents, state_dim, device=device)
        elif config.mixer == 'qtrans':
            action_dim = space2shape(action_space)
            self.mixer = QTRAN_mixer(n_agents, state_dim, action_dim, device=device)
        else:
            self.mixer = VDN_mixer(device=device)

    def forward(self, observations: Dict[str, torch.Tensor]):
        qs = []
        for key in self.agent_keys:
            q = self.agents[key](observations[key])
            qs.append(q)
        return torch.stack(qs, dim=1)  # [batch, n_agents, n_actions]

    def Qvalues(self, observations):
        return self(observations)

    def sample_actions(self, q_values, epsilon=0.0):
        q_values = q_values.cpu().numpy()
        if epsilon > 0 and np.random.random() < epsilon:
            return {key: np.random.randint(0, q_values.shape[-1], size=q_values.shape[0])
                    for key in self.agent_keys}
        actions = q_values.argmax(axis=-1)
        return {key: actions[..., i] for i, key in enumerate(self.agent_keys)}


class Weighted_MixingQnetwork(MixingQnetwork):
    """Q-network with WQMIX mixing."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str], device=None):
        super().__init__(observation_space, action_space, config, n_agents, agent_keys, device)
        state_dim = config.state_space if isinstance(config.state_space, int) else space2shape(config.state_space)
        self.mixer = WQMIX_mixer(n_agents, state_dim, device=device)


class Independent_DDPG_Policy(nn.Module):
    """Independent DDPG policy (one actor + critic per agent, no sharing)."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str], device=None):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = agent_keys
        self.config = config
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        obs_dim = space2shape(observation_space)
        act_dim = space2shape(action_space)

        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        for key in agent_keys:
            self.actors[key] = nn.Sequential(
                nn.Linear(obs_dim, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, act_dim, device=device),
                nn.Tanh(),
            )
            self.critics[key] = nn.Sequential(
                nn.Linear(obs_dim + act_dim, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, 1, device=device),
            )

    def forward(self, observations: Dict[str, torch.Tensor]):
        actions = {}
        for key in self.agent_keys:
            actions[key] = self.actors[key](observations[key])
        return actions

    def Qvalues(self, observations, actions):
        qs = []
        for key in self.agent_keys:
            x = torch.cat([observations[key], actions[key]], dim=-1)
            qs.append(self.critics[key](x))
        return torch.stack(qs, dim=-1)

    def sample_actions(self, observations, epsilon=0.0):
        with torch.no_grad():
            actions = self(observations)
            actions = {k: v.cpu().numpy() for k, v in actions.items()}
            if epsilon > 0:
                for k in actions.keys():
                    noise = np.random.normal(0, epsilon, size=actions[k].shape)
                    actions[k] = (actions[k] + noise).clip(-1, 1)
            act_space = getattr(self, 'action_space', None)
            if act_space is not None and hasattr(act_space, 'n'):
                actions = {k: v.argmax(axis=-1) for k, v in actions.items()}
        return actions


class MADDPG_Policy(nn.Module):
    """MADDPG policy: centralized critic with per-agent actor."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str], device=None):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = agent_keys
        self.config = config
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        obs_dim = space2shape(observation_space)
        act_dim = space2shape(action_space)

        self.actors = nn.ModuleDict()
        for key in agent_keys:
            self.actors[key] = nn.Sequential(
                nn.Linear(obs_dim, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, act_dim, device=device),
                nn.Tanh(),
            )

        # Centralized critic: takes all observations + all actions
        self.critic = nn.Sequential(
            nn.Linear(n_agents * obs_dim + n_agents * act_dim, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 1, device=device),
        )

    def forward(self, observations: Dict[str, torch.Tensor]):
        actions = {}
        for key in self.agent_keys:
            actions[key] = self.actors[key](observations[key])
        return actions

    def Qvalues(self, observations, actions):
        obs_all = torch.cat([observations[k] for k in self.agent_keys], dim=-1)
        act_all = torch.cat([actions[k] for k in self.agent_keys], dim=-1)
        return self.critic(torch.cat([obs_all, act_all], dim=-1))

    def sample_actions(self, observations, epsilon=0.0):
        with torch.no_grad():
            actions = self(observations)
            actions = {k: v.cpu().numpy() for k, v in actions.items()}
            if epsilon > 0:
                for k in actions.keys():
                    noise = np.random.normal(0, epsilon, size=actions[k].shape)
                    actions[k] = (actions[k] + noise).clip(-1, 1)
            # Handle discrete action spaces: convert continuous to argmax
            act_space = getattr(self, 'action_space', None)
            if act_space is not None and hasattr(act_space, 'n'):
                actions = {k: v.argmax(axis=-1) for k, v in actions.items()}
        return actions


class Categorical_MAAC_Policy(nn.Module):
    """Categorical actor for IPPO/MAPPO (discrete actions)."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str],
                 use_rnn=False, device=None):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = agent_keys
        self.config = config
        self.device = device
        self.use_rnn = use_rnn
        self.observation_space = observation_space
        self.action_space = action_space
        obs_dim = space2shape(observation_space)
        self.action_dim = space2shape(action_space)

        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        for key in agent_keys:
            self.actors[key] = nn.Sequential(
                nn.Linear(obs_dim, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, self.action_dim, device=device),
            )
            if use_rnn:
                self.critics[key] = CriticNet_RNN(obs_dim, device=device)
            else:
                self.critics[key] = nn.Sequential(
                    nn.Linear(obs_dim, 64, device=device),
                    nn.ReLU(),
                    nn.Linear(64, 1, device=device),
                )

    def forward(self, observations: Dict[str, torch.Tensor]):
        logits = {}
        for key in self.agent_keys:
            logits[key] = self.actors[key](observations[key])
        return logits

    def get_value(self, observations: Dict[str, torch.Tensor]):
        values = {}
        for key in self.agent_keys:
            if self.use_rnn:
                v, _ = self.critics[key](observations[key])
                values[key] = v
            else:
                values[key] = self.critics[key](observations[key])
        return values

    def sample_actions(self, logits, epsilon=0.0):
        actions = {}
        for key in self.agent_keys:
            l = logits[key]
            if epsilon > 0 and np.random.random() < epsilon:
                actions[key] = np.random.randint(0, self.action_dim,
                                                  size=l.shape[0])
            else:
                probs = F.softmax(l, dim=-1)
                actions[key] = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
        return actions


class Gaussian_MAAC_Policy(nn.Module):
    """Gaussian actor for IPPO/MAPPO (continuous actions)."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str],
                 use_rnn=False, device=None):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = agent_keys
        self.config = config
        self.device = device
        self.use_rnn = use_rnn
        self.observation_space = observation_space
        self.action_space = action_space
        obs_dim = space2shape(observation_space)
        self.action_dim = space2shape(action_space)

        self.actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        for key in agent_keys:
            self.actors[key] = GaussianActorNet(obs_dim, self.action_dim, device=device)
            if use_rnn:
                self.critics[key] = CriticNet_RNN(obs_dim, device=device)
            else:
                self.critics[key] = CriticNet(obs_dim, device=device)

    def forward(self, observations: Dict[str, torch.Tensor]):
        mus = {}
        stds = {}
        for key in self.agent_keys:
            mu, std = self.actors[key](observations[key])
            mus[key] = mu
            stds[key] = std
        return mus, stds

    def get_value(self, observations: Dict[str, torch.Tensor]):
        values = {}
        for key in self.agent_keys:
            if self.use_rnn:
                v, _ = self.critics[key](observations[key])
                values[key] = v
            else:
                values[key] = self.critics[key](observations[key])
        return values

    def sample_actions(self, observations, epsilon=0.0, explored=True):
        actions = {}
        with torch.no_grad():
            mus, stds = self(observations)
            for key in self.agent_keys:
                if explored:
                    dist = torch.distributions.Normal(mus[key], stds[key])
                    a = dist.sample()
                else:
                    a = mus[key]
                actions[key] = a.cpu().numpy()
        return actions


class COMA_Policy(nn.Module):
    """COMA: counterfactual multi-agent policy gradients."""
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 config: CArgs, n_agents: int, agent_keys: List[str], device=None):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = agent_keys
        self.config = config
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space
        obs_dim = space2shape(observation_space)
        self.action_dim = space2shape(action_space)

        # Per-agent actors with action masking
        self.actors = nn.ModuleDict()
        for key in agent_keys:
            self.actors[key] = nn.Sequential(
                nn.Linear(obs_dim, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, self.action_dim, device=device),
            )

        # Centralized critic: obs of all agents + actions of all agents
        state_dim = (n_agents * obs_dim) + (n_agents * self.action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, self.action_dim, device=device),  # Q values per action per agent
        )

    def forward(self, observations: Dict[str, torch.Tensor]):
        logits = {}
        for key in self.agent_keys:
            logits[key] = self.actors[key](observations[key])
        return logits

    def Qvalues(self, observations, actions):
        obs_all = torch.cat([observations[k] for k in self.agent_keys], dim=-1)
        # Convert scalar actions to one-hot
        act_onehots = []
        for k in self.agent_keys:
            a = actions[k]
            if a.dim() == 1 or a.shape[-1] == 1:
                a = F.one_hot(a.long().squeeze(-1), self.action_dim).float()
            act_onehots.append(a)
        act_all = torch.cat(act_onehots, dim=-1)
        q_all = self.critic(torch.cat([obs_all, act_all], dim=-1))
        qs = {}
        for i, key in enumerate(self.agent_keys):
            qs[key] = q_all  # Same Q for all agents (centralized)
        return qs

    def sample_actions(self, logits, epsilon=0.0):
        actions = {}
        for key in self.agent_keys:
            l = logits[key]
            if epsilon > 0 and np.random.random() < epsilon:
                actions[key] = np.random.randint(0, self.action_dim,
                                                  size=l.shape[0])
            else:
                probs = F.softmax(l, dim=-1)
                actions[key] = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
        return actions

# ──────────────────────────────────────────────
# Replay Buffers
# ──────────────────────────────────────────────

class MARL_OffPolicyBuffer:
    """Replay buffer for off-policy MARL algorithms."""
    def __init__(self, buffer_size: int, agent_keys: List[str],
                 obs_dim: int, act_dim: int, state_dim: int,
                 n_agents: int, device=None):
        self.buffer_size = buffer_size
        self.agent_keys = agent_keys
        self.n_agents = n_agents
        self.device = device
        self.pos = 0
        self.full = False

        self.obs = {k: np.zeros((buffer_size, obs_dim), dtype=np.float32) for k in agent_keys}
        self.obs_next = {k: np.zeros((buffer_size, obs_dim), dtype=np.float32) for k in agent_keys}
        self.actions = {k: np.zeros((buffer_size, act_dim), dtype=np.float32) if isinstance(act_dim, int) and act_dim > 1 else np.zeros((buffer_size,), dtype=np.int64) for k in agent_keys}
        self.rewards = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.dones = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.terminals = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.state_next = np.zeros((buffer_size, state_dim), dtype=np.float32)

    def store(self, step_data):
        p = self.pos % self.buffer_size
        for k in self.agent_keys:
            self.obs[k][p] = step_data['obs'][k]
            self.obs_next[k][p] = step_data['obs_next'][k]
            self.actions[k][p] = step_data['actions'][k]
            self.rewards[k][p] = step_data['rewards'][k]
            self.dones[k][p] = step_data['dones'][k]
            self.terminals[k][p] = step_data['terminals'][k]
        self.state[p] = step_data['state']
        self.state_next[p] = step_data['state_next']
        self.pos = p + 1
        if not self.full and self.pos >= self.buffer_size:
            self.full = True

    def sample(self, batch_size: int):
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)
        batch = {
            'obs': {k: torch.FloatTensor(self.obs[k][indices]).to(self.device) for k in self.agent_keys},
            'obs_next': {k: torch.FloatTensor(self.obs_next[k][indices]).to(self.device) for k in self.agent_keys},
            'actions': {k: torch.FloatTensor(self.actions[k][indices]).to(self.device) if self.actions[k][indices].dtype == np.float32 else torch.LongTensor(self.actions[k][indices]).to(self.device) for k in self.agent_keys},
            'rewards': {k: torch.FloatTensor(self.rewards[k][indices]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'dones': {k: torch.FloatTensor(self.dones[k][indices]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'terminals': {k: torch.FloatTensor(self.terminals[k][indices]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'state': torch.FloatTensor(self.state[indices]).to(self.device),
            'state_next': torch.FloatTensor(self.state_next[indices]).to(self.device),
        }
        return batch

    def __len__(self):
        return self.buffer_size if self.full else self.pos


class MARL_OnPolicyBuffer:
    """Buffer for on-policy MARL algorithms (PPO, COMA)."""
    def __init__(self, buffer_size: int, agent_keys: List[str],
                 obs_dim: int, act_dim: int, state_dim: int,
                 n_agents: int, device=None):
        self.buffer_size = buffer_size
        self.agent_keys = agent_keys
        self.n_agents = n_agents
        self.device = device
        self.ptr = 0

        self.obs = {k: np.zeros((buffer_size, obs_dim), dtype=np.float32) for k in agent_keys}
        self.actions = {k: np.zeros((buffer_size,), dtype=np.int64) for k in agent_keys}
        self.rewards = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.dones = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.log_probs = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.values = {k: np.zeros((buffer_size,), dtype=np.float32) for k in agent_keys}
        self.state = np.zeros((buffer_size, state_dim), dtype=np.float32)

    def store(self, step_data):
        p = self.ptr
        for k in self.agent_keys:
            self.obs[k][p] = step_data['obs'][k]
            self.actions[k][p] = step_data['actions'][k]
            self.rewards[k][p] = step_data['rewards'][k]
            self.dones[k][p] = step_data['dones'][k]
            if 'log_probs' in step_data and k in step_data['log_probs']:
                self.log_probs[k][p] = step_data['log_probs'][k]
            if 'values' in step_data and k in step_data['values']:
                self.values[k][p] = step_data['values'][k]
        self.state[p] = step_data['state']
        self.ptr = p + 1

    def clear(self):
        self.ptr = 0

    def get(self):
        batch = {
            'obs': {k: torch.FloatTensor(self.obs[k][:self.ptr]).to(self.device) for k in self.agent_keys},
            'actions': {k: torch.LongTensor(self.actions[k][:self.ptr]).to(self.device) for k in self.agent_keys},
            'rewards': {k: torch.FloatTensor(self.rewards[k][:self.ptr]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'dones': {k: torch.FloatTensor(self.dones[k][:self.ptr]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'log_probs': {k: torch.FloatTensor(self.log_probs[k][:self.ptr]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'values': {k: torch.FloatTensor(self.values[k][:self.ptr]).unsqueeze(-1).to(self.device) for k in self.agent_keys},
            'state': torch.FloatTensor(self.state[:self.ptr]).to(self.device),
        }
        return batch

    def __len__(self):
        return self.ptr

# ──────────────────────────────────────────────
# Learners
# ──────────────────────────────────────────────

class IQL_Learner:
    """Independent Q-Learning learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.sync_frequency = config.sync_frequency if hasattr(config, 'sync_frequency') else 100
        self.tau = config.tau if hasattr(config, 'tau') else 0.005

        self.target_policy = _copy_policy(policy, device)
        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs_next = batch['obs_next']

        # Get current Q values
        q_values = self.policy.Qvalues(obs)
        # Handle both dict (multi-agent) and single cases
        if isinstance(q_values, dict):
            total_loss = 0.0
            for k in q_values.keys():
                q = q_values[k]
                a = actions[k].long().unsqueeze(-1)
                q_taken = q.gather(-1, a).squeeze(-1)
                with torch.no_grad():
                    q_next = self.target_policy.Qvalues({k: obs_next[k]})
                    if isinstance(q_next, dict):
                        q_next = q_next[k]
                    q_target = rewards[k].squeeze(-1) + self.gamma * (1 - terminals[k].squeeze(-1)) * q_next.max(dim=-1)[0]
                loss = F.mse_loss(q_taken, q_target)
                total_loss += loss
            loss = total_loss / len(q_values)
        else:
            q = q_values
            a = actions.long().unsqueeze(-1)
            q_taken = q.gather(-1, a).squeeze(-1)
            with torch.no_grad():
                q_next = self.target_policy.Qvalues(obs_next)
                q_target = rewards.squeeze(-1) + self.gamma * (1 - terminals.squeeze(-1)) * q_next.max(dim=-1)[0]
            loss = F.mse_loss(q_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_target(self):
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def _copy_policy(policy, device):
    """Create a target policy copy without deepcopy issues."""
    sd = {k: v.clone() for k, v in policy.state_dict().items()}
    cls = type(policy)

    obs_space = getattr(policy, 'observation_space', None)
    act_space = getattr(policy, 'action_space', None)
    config = getattr(policy, 'config', CArgs())
    n_agents = getattr(policy, 'n_agents', 1)
    agent_keys = getattr(policy, 'agent_keys', [])
    use_rnn = getattr(policy, 'use_rnn', False)

    cls_name = cls.__name__
    if cls_name == 'BasicQnetwork_marl':
        result = cls(obs_space, act_space, config, device=device)
    elif 'MAAC' in cls_name:
        result = cls(obs_space, act_space, config, n_agents, agent_keys, use_rnn, device=device)
    else:
        result = cls(obs_space, act_space, config, n_agents, agent_keys, device=device)
    result.load_state_dict(sd)
    return result


class VDN_Learner:
    """Value Decomposition Network learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.sync_frequency = config.sync_frequency if hasattr(config, 'sync_frequency') else 100
        self.tau = config.tau if hasattr(config, 'tau') else 0.005

        self.target_policy = _copy_policy(policy, device)
        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs_next = batch['obs_next']

        # Get per-agent Q values
        q_values = self.policy.Qvalues(obs)  # [batch, n_agents, n_actions] or dict
        # Convert dict to tensor
        if isinstance(q_values, dict):
            q_list = []
            act_list = []
            for k in self.policy.agent_keys:
                q_list.append(q_values[k])
                act_list.append(actions[k].long())
            q_values = torch.stack(q_list, dim=1)  # [batch, n_agents, n_actions]
            actions_stack = torch.stack(act_list, dim=1)  # [batch, n_agents]
        else:
            actions_stack = torch.stack([actions[k].long() for k in self.policy.agent_keys], dim=1)

        # Q_tot via mixing
        q_taken = q_values.gather(-1, actions_stack.unsqueeze(-1)).squeeze(-1)
        q_tot = self.policy.mixer(q_taken, None)  # VDN: sum, shape [batch, 1]

        # Target Q_tot
        with torch.no_grad():
            q_next = self.target_policy.Qvalues(obs_next)
            if isinstance(q_next, dict):
                q_next_list = []
                for k in self.policy.agent_keys:
                    q_next_list.append(q_next[k])
                q_next = torch.stack(q_next_list, dim=1)
            q_next_max = q_next.max(dim=-1)[0]  # [batch, n_agents]
            q_tot_next = self.target_policy.mixer(q_next_max, None)

        rewards_mean = torch.stack([rewards[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1, keepdim=True)
        terminals_mean = torch.stack([terminals[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1, keepdim=True)
        q_target = rewards_mean + self.gamma * (1 - terminals_mean) * q_tot_next

        loss = F.mse_loss(q_tot, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_target(self):
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class QMIX_Learner:
    """QMIX learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.sync_frequency = config.sync_frequency if hasattr(config, 'sync_frequency') else 100
        self.tau = config.tau if hasattr(config, 'tau') else 0.005

        self.target_policy = _copy_policy(policy, device)
        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs_next = batch['obs_next']
        state = batch['state']
        state_next = batch['state_next']

        q_values = self.policy.Qvalues(obs)
        if isinstance(q_values, dict):
            q_list = []
            act_list = []
            for k in self.policy.agent_keys:
                q_list.append(q_values[k])
                act_list.append(actions[k].long())
            q_values = torch.stack(q_list, dim=1)
            actions_stack = torch.stack(act_list, dim=1)
        else:
            actions_stack = torch.stack([actions[k].long() for k in self.policy.agent_keys], dim=1)

        q_taken = q_values.gather(-1, actions_stack.unsqueeze(-1)).squeeze(-1)
        q_tot = self.policy.mixer(q_taken, state)  # [batch, 1]

        with torch.no_grad():
            q_next = self.target_policy.Qvalues(obs_next)
            if isinstance(q_next, dict):
                q_next_list = []
                for k in self.policy.agent_keys:
                    q_next_list.append(q_next[k])
                q_next = torch.stack(q_next_list, dim=1)
            q_next_max = q_next.max(dim=-1)[0]  # [batch, n_agents]
            q_tot_next = self.target_policy.mixer(q_next_max, state_next)

        rewards_mean = torch.stack([rewards[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1)
        terminals_mean = torch.stack([terminals[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1)
        q_target = rewards_mean + self.gamma * (1 - terminals_mean) * q_tot_next.squeeze(-1)

        loss = F.mse_loss(q_tot.squeeze(-1), q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_target(self):
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class WQMIX_Learner:
    """Weighted QMIX learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.sync_frequency = config.sync_frequency if hasattr(config, 'sync_frequency') else 100
        self.tau = config.tau if hasattr(config, 'tau') else 0.005
        self.alpha = config.alpha if hasattr(config, 'alpha') else 0.5

        self.target_policy = _copy_policy(policy, device)
        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs_next = batch['obs_next']
        state = batch['state']
        state_next = batch['state_next']

        q_values = self.policy.Qvalues(obs)
        if isinstance(q_values, dict):
            q_list = []
            act_list = []
            for k in self.policy.agent_keys:
                q_list.append(q_values[k])
                act_list.append(actions[k].long())
            q_values = torch.stack(q_list, dim=1)
            actions_stack = torch.stack(act_list, dim=1)
        else:
            actions_stack = torch.stack([actions[k].long() for k in self.policy.agent_keys], dim=1)

        q_taken = q_values.gather(-1, actions_stack.unsqueeze(-1)).squeeze(-1)
        q_tot, q_tot_opt, weight_net = self.policy.mixer(q_taken, state)

        with torch.no_grad():
            q_next = self.target_policy.Qvalues(obs_next)
            if isinstance(q_next, dict):
                q_next_list = []
                for k in self.policy.agent_keys:
                    q_next_list.append(q_next[k])
                q_next = torch.stack(q_next_list, dim=1)
            q_next_max = q_next.max(dim=-1, keepdim=True)[0]
            q_tot_next, q_tot_opt_next, _ = self.target_policy.mixer(
                q_next_max.squeeze(-1), state_next)

        rewards_mean = torch.stack([rewards[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1)
        terminals_mean = torch.stack([terminals[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1)
        q_target = rewards_mean + self.gamma * (1 - terminals_mean) * q_tot_next.squeeze(-1)
        q_opt_target = rewards_mean + self.gamma * (1 - terminals_mean) * q_tot_opt_next.squeeze(-1)

        # Compute weights
        w = torch.where(q_tot_opt.squeeze(-1) > q_tot.squeeze(-1) + 1e-8,
                        torch.ones_like(q_tot.squeeze(-1)) * self.alpha,
                        torch.ones_like(q_tot.squeeze(-1)))
        w = w.detach()

        # QMIX loss (weighted)
        td_error = q_tot.squeeze(-1) - q_target.detach()
        loss_qmix = (w * td_error ** 2).mean()

        # Opt loss
        td_opt = q_tot_opt.squeeze(-1) - q_opt_target.detach()
        loss_opt = (td_opt ** 2).mean()

        loss = loss_qmix + loss_opt

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        return {'loss': loss.item(), 'loss_qmix': loss_qmix.item(), 'loss_opt': loss_opt.item()}

    def update_target(self):
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class MADDPG_Learner:
    """MADDPG learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.95
        self.tau = config.tau if hasattr(config, 'tau') else 0.005

        self.target_policy = _copy_policy(policy, device)

        self.actor_optimizer = optimizer or torch.optim.Adam(
            policy.actors.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)
        self.critic_optimizer = torch.optim.Adam(
            policy.critic.parameters(), lr=config.critic_learning_rate if hasattr(config, 'critic_learning_rate') else 1e-3)

    def update(self, batch):
        obs = batch['obs']
        obs_next = batch['obs_next']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']

        # Centralized critic update
        with torch.no_grad():
            target_actions = self.target_policy(obs_next)
            q_next = self.target_policy.Qvalues(obs_next, target_actions)
        rewards_mean = torch.stack([rewards[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1, keepdim=True)
        terminals_mean = torch.stack([terminals[k].squeeze(-1) for k in self.policy.agent_keys], dim=1).mean(dim=1, keepdim=True)
        q_target = rewards_mean + self.gamma * (1 - terminals_mean) * q_next

        q_current = self.policy.Qvalues(obs, actions)
        critic_loss = F.mse_loss(q_current, q_target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Actor update
        current_actions = self.policy(obs)
        q_actors = self.policy.Qvalues(obs, current_actions)
        actor_loss = -q_actors.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actors.parameters(), 10.0)
        self.actor_optimizer.step()

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

    def update_target(self):
        for target_param, param in zip(self.target_policy.actors.parameters(), self.policy.actors.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_policy.critic.parameters(), self.policy.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class IDDPG_Learner:
    """Independent DDPG learner (per-agent actor-critic)."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.95
        self.tau = config.tau if hasattr(config, 'tau') else 0.005

        self.target_policy = _copy_policy(policy, device)

        self.actor_optimizer = optimizer or torch.optim.Adam(
            policy.actors.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)
        self.critic_optimizer = torch.optim.Adam(
            policy.critics.parameters(), lr=config.critic_learning_rate if hasattr(config, 'critic_learning_rate') else 1e-3)

    def update(self, batch):
        obs = batch['obs']
        obs_next = batch['obs_next']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']

        # Critic update per agent
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        for key in self.policy.agent_keys:
            with torch.no_grad():
                target_act = self.target_policy.actors[key](obs_next[key])
                q_next = self.target_policy.critics[key](
                    torch.cat([obs_next[key], target_act], dim=-1))
            q_target = rewards[key] + self.gamma * (1 - terminals[key]) * q_next
            q_current = self.policy.critics[key](
                torch.cat([obs[key], actions[key]], dim=-1))
            critic_loss = F.mse_loss(q_current, q_target.detach())
            total_critic_loss += critic_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critics.parameters(), 10.0)
        self.critic_optimizer.step()

        # Actor update per agent
        for key in self.policy.agent_keys:
            current_act = self.policy.actors[key](obs[key])
            q = self.policy.critics[key](
                torch.cat([obs[key], current_act], dim=-1))
            actor_loss = -q.mean()
            total_actor_loss += actor_loss

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actors.parameters(), 10.0)
        self.actor_optimizer.step()

        return {
            'critic_loss': (total_critic_loss / len(self.policy.agent_keys)).item(),
            'actor_loss': (total_actor_loss / len(self.policy.agent_keys)).item(),
        }

    def update_target(self):
        for target_param, param in zip(self.target_policy.actors.parameters(), self.policy.actors.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_policy.critics.parameters(), self.policy.critics.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class IPPO_Learner:
    """Independent PPO learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.gae_lambda = config.gae_lambda if hasattr(config, 'gae_lambda') else 0.95
        self.clip_range = config.clip_range if hasattr(config, 'clip_range') else 0.2
        self.entropy_coef = config.entropy_coef if hasattr(config, 'entropy_coef') else 0.01
        self.value_coef = config.value_coef if hasattr(config, 'value_coef') else 0.5
        self.max_grad_norm = config.max_grad_norm if hasattr(config, 'max_grad_norm') else 0.5
        self.epochs = config.epochs if hasattr(config, 'epochs') else 10
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 64

        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 3e-4)

    def compute_gae(self, rewards, values, dones):
        """Compute GAE for a single agent."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
        returns = advantages + values
        return advantages, returns

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        old_log_probs = batch['log_probs']
        old_values = batch['values']
        state = batch.get('state')

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for key in self.policy.agent_keys:
            # Compute advantages and returns
            advantages, returns = self.compute_gae(
                rewards[key], old_values[key], dones[key])

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = obs[key].shape[0]
            for _ in range(self.epochs):
                indices = torch.randperm(dataset_size)
                for start in range(0, dataset_size, self.batch_size):
                    idx = indices[start:start + self.batch_size]

                    batch_obs = obs[key][idx]
                    batch_actions = actions[key][idx]
                    batch_advantages = advantages[idx]
                    batch_returns = returns[idx]
                    batch_old_log_probs = old_log_probs[key][idx]

                    # Get new log probs and values
                    logits = self.policy.actors[key](batch_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions.squeeze(-1))
                    entropy = dist.entropy().mean()

                    new_values = self.policy.critics[key](batch_obs)

                    # Policy loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze(-1))
                    surr1 = ratio * batch_advantages.squeeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages.squeeze(-1)
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(new_values.squeeze(-1), batch_returns.squeeze(-1))

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()

        n_updates = len(self.policy.agent_keys) * self.epochs * (dataset_size // self.batch_size + 1)
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def update_target(self):
        pass  # PPO does not use target networks


class MAPPO_Learner:
    """Multi-Agent PPO learner (centralized critic)."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.gae_lambda = config.gae_lambda if hasattr(config, 'gae_lambda') else 0.95
        self.clip_range = config.clip_range if hasattr(config, 'clip_range') else 0.2
        self.entropy_coef = config.entropy_coef if hasattr(config, 'entropy_coef') else 0.01
        self.value_coef = config.value_coef if hasattr(config, 'value_coef') else 0.5
        self.max_grad_norm = config.max_grad_norm if hasattr(config, 'max_grad_norm') else 0.5
        self.epochs = config.epochs if hasattr(config, 'epochs') else 10
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 64

        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 3e-4)

    def compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
        returns = advantages + values
        return advantages, returns

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        old_log_probs = batch['log_probs']
        old_values = batch['values']
        state = batch.get('state')

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for key in self.policy.agent_keys:
            advantages, returns = self.compute_gae(
                rewards[key], old_values[key], dones[key])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = obs[key].shape[0]
            for _ in range(self.epochs):
                indices = torch.randperm(dataset_size)
                for start in range(0, dataset_size, self.batch_size):
                    idx = indices[start:start + self.batch_size]

                    batch_obs = obs[key][idx]
                    batch_actions = actions[key][idx]
                    batch_advantages = advantages[idx]
                    batch_returns = returns[idx]
                    batch_old_log_probs = old_log_probs[key][idx]

                    logits = self.policy.actors[key](batch_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions.squeeze(-1))
                    entropy = dist.entropy().mean()

                    # MAPPO: centralized critic uses global state
                    if state is not None:
                        new_values = self.policy.critics[key](state[idx])
                    else:
                        new_values = self.policy.critics[key](batch_obs)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs.squeeze(-1))
                    surr1 = ratio * batch_advantages.squeeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages.squeeze(-1)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_values.squeeze(-1), batch_returns.squeeze(-1))
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()

        n_updates = len(self.policy.agent_keys) * self.epochs * (dataset_size // self.batch_size + 1)
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def update_target(self):
        pass


class COMA_Learner:
    """COMA learner."""
    def __init__(self, policy, config: CArgs, optimizer=None, device=None):
        self.policy = policy
        self.config = config
        self.device = device
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.99
        self.entropy_coef = config.entropy_coef if hasattr(config, 'entropy_coef') else 0.01
        self.max_grad_norm = config.max_grad_norm if hasattr(config, 'max_grad_norm') else 10.0

        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate if hasattr(config, 'learning_rate') else 1e-3)

    def compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(rewards.shape[0])):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns

    def update(self, batch):
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']

        total_loss = 0.0
        for key in self.policy.agent_keys:
            # Compute returns
            returns = self.compute_returns(rewards[key], dones[key])

            # Get log probs
            logits = self.policy.actors[key](obs[key])
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions[key].squeeze(-1))
            entropy = dist.entropy().mean()

            # COMA: Q-values from centralized critic, counterfactual baseline
            q_values = self.policy.Qvalues(obs, actions)[key]  # [batch, n_actions]

            # Baseline: expected Q under marginal action distribution
            probs = F.softmax(logits, dim=-1)
            baseline = (probs * q_values).sum(dim=-1)

            # Advantage
            q_taken = q_values.gather(-1, actions[key].long().unsqueeze(-1)).squeeze(-1)
            advantages = q_taken - baseline.detach()

            # Normalize
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.detach()

            # Policy loss
            policy_loss = -(log_probs * advantages.squeeze(-1)).mean()
            loss = policy_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / len(self.policy.agent_keys)}

    def update_target(self):
        pass

# ──────────────────────────────────────────────
# Environments
# ──────────────────────────────────────────────

class MPE_Env:
    """Wrapper for MPE (Simple Spread v3)."""
    def __init__(self, env_id: str, seed: int = 0, continuous_actions: bool = False):
        self.env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25,
                                                  continuous_actions=continuous_actions)
        self.env.reset(seed=seed)
        self.agents = self.env.possible_agents
        self.agent_keys = self.agents
        self.n_agents = len(self.agents)
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])
        self.continuous_actions = continuous_actions
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents * space2shape(self.observation_space),))
        self.max_episode_steps = 25

    def reset(self):
        obs, infos = self.env.reset()
        state = self._get_state(obs)
        return obs, state, infos

    def step(self, actions):
        obs_next, rewards, terms, truncs, infos = self.env.step(actions)
        dones = {k: terms.get(k, False) or truncs.get(k, False) for k in self.agent_keys}
        state_next = self._get_state(obs_next)
        return obs_next, rewards, dones, terms, truncs, state_next, infos

    def _get_state(self, obs):
        return np.concatenate([obs[k] for k in self.agent_keys], axis=0).astype(np.float32)

    def close(self):
        self.env.close()


class RoboticWarehouseEnv:
    """Wrapper for rware environment using gym.make.

    Converts between rware's tuple-based interface
    (Tuple(Discrete(5), ...) / Tuple(Box(71,), ...))
    and the dict-based interface used throughout algorithm.py.
    """
    def __init__(self, env_id: str, seed: int = 0):
        import gymnasium as gym
        import rware
        self._env = gym.make(env_id)
        self._env.reset(seed=seed)
        self.agents = [f'agent_{i}' for i in range(self._env.unwrapped.n_agents)]
        self.agent_keys = self.agents
        self.n_agents = len(self.agents)
        raw_obs_space = self._env.observation_space[0]  # Box(71,)
        raw_act_space = self._env.action_space[0]        # Discrete(5)
        self.observation_space = raw_obs_space
        self.action_space = raw_act_space
        obs_dim = space2shape(raw_obs_space)
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents * obs_dim,))
        self.max_episode_steps = getattr(self._env.unwrapped, 'max_steps', 500)

    def reset(self):
        obs_tuple, info = self._env.reset()
        obs = {k: np.array(v, dtype=np.float32) for k, v in zip(self.agent_keys, obs_tuple)}
        state = self._get_state(obs)
        return obs, state, info

    def step(self, actions):
        actions_list = [int(actions[k]) for k in self.agent_keys]
        result = self._env.step(actions_list)
        # rware returns (obs, rewards, terminated, truncated, info)
        # where terminated/truncated are single bools (not per-agent lists)
        obs_tuple, rewards_list, terminated, truncated, infos = result
        obs_next = {k: np.array(v, dtype=np.float32) for k, v in zip(self.agent_keys, obs_tuple)}
        rewards = {k: float(r) for k, r in zip(self.agent_keys, rewards_list)}
        terms = {k: bool(terminated) for k in self.agent_keys}
        truncs = {k: bool(truncated) for k in self.agent_keys}
        dones = {k: terms[k] or truncs[k] for k in self.agent_keys}
        state_next = self._get_state(obs_next)
        return obs_next, rewards, dones, terms, truncs, state_next, infos

    def _get_state(self, obs):
        return np.concatenate([obs[k] for k in self.agent_keys], axis=0).astype(np.float32)

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


class DummyVecMultiAgentEnv:
    """Vectorized wrapper for a single multi-agent environment."""
    def __init__(self, env_fn):
        self.env = env_fn()
        self.agents = self.env.agents
        self.agent_keys = self.env.agent_keys
        self.n_agents = self.env.n_agents
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_space = self.env.state_space
        self.max_episode_steps = self.env.max_episode_steps
        self.num_envs = 1
        self.episode_steps = 0

    def reset(self):
        self.episode_steps = 0
        obs, state, infos = self.env.reset()
        return obs, state, infos

    def step(self, actions):
        self.episode_steps += 1
        obs, rewards, dones, terms, truncs, state, infos = self.env.step(actions)
        return obs, rewards, dones, terms, truncs, state, infos

    def close(self):
        self.env.close()


CONTINUOUS_ALGOS = {'maddpg', 'iddpg'}

def _get_env(env_id: str, algo: str = ''):
    """Create an environment instance based on env_id."""
    continuous = algo.lower().replace('_', '') in CONTINUOUS_ALGOS
    if 'rware' in env_id.lower() or 'robotic_warehouse' in env_id.lower():
        return RoboticWarehouseEnv(env_id)
    elif 'simple_spread' in env_id.lower() or 'mpe' in env_id.lower():
        return MPE_Env(env_id, continuous_actions=continuous)
    else:
        raise ValueError(f"Unknown environment: {env_id}")


# ──────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────

class MARLAgents:
    """Base class for all MARL agents."""
    def __init__(self, policy, learner, config: CArgs, device=None):
        self.policy = policy
        self.learner = learner
        self.config = config
        self.device = device

    def act(self, obs, epsilon=0.0):
        raise NotImplementedError

    def train(self):
        self.policy.train()
        self.learner.policy.train()

    def eval(self):
        self.policy.eval()
        self.learner.policy.eval()


class OffPolicyMARLAgents(MARLAgents):
    """Base for off-policy MARL agents."""
    def __init__(self, policy, learner, buffer, config: CArgs, device=None):
        super().__init__(policy, learner, config, device)
        self.buffer = buffer
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 32

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            q_values = self.policy.Qvalues(obs_tensors)
        actions = self.policy.sample_actions(q_values, epsilon=epsilon)
        return {k: v.item() if isinstance(v, np.ndarray) else v for k, v in actions.items()}

    def store(self, step_data):
        self.buffer.store(step_data)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)
        info = self.learner.update(batch)
        self.learner.update_target()
        return info


class OnPolicyMARLAgents(MARLAgents):
    """Base for on-policy MARL agents."""
    def __init__(self, policy, learner, buffer, config: CArgs, device=None):
        super().__init__(policy, learner, config, device)
        self.buffer = buffer
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 64

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            logits = self.policy(obs_tensors)
        return self.policy.sample_actions(logits, epsilon=epsilon)

    def store(self, step_data):
        self.buffer.store(step_data)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.get()
        info = self.learner.update(batch)
        self.buffer.clear()
        return info


class IQL_Agents(OffPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        policy = BasicQnetwork_marl(env.observation_space, env.action_space, config, device)
        learner = IQL_Learner(policy, config, device=device)
        buffer = MARL_OffPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 10000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)


class VDN_Agents(OffPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        # Override mixer to 'vdn'
        config.mixer = 'vdn'
        config.state_space = env.state_space

        policy = MixingQnetwork(env.observation_space, env.action_space, config,
                                self.n_agents, self.agent_keys, device)
        learner = VDN_Learner(policy, config, device=device)
        buffer = MARL_OffPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 10000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)


class QMIX_Agents(OffPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        config.mixer = 'qmix'
        config.state_space = env.state_space

        policy = MixingQnetwork(env.observation_space, env.action_space, config,
                                self.n_agents, self.agent_keys, device)
        learner = QMIX_Learner(policy, config, device=device)
        buffer = MARL_OffPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 10000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)


class WQMIX_Agents(OffPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        config.mixer = 'wqmix'
        config.state_space = env.state_space

        policy = Weighted_MixingQnetwork(env.observation_space, env.action_space, config,
                                          self.n_agents, self.agent_keys, device)
        learner = WQMIX_Learner(policy, config, device=device)
        buffer = MARL_OffPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 10000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)


class MADDPG_Agents(OffPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        policy = MADDPG_Policy(env.observation_space, env.action_space, config,
                                self.n_agents, self.agent_keys, device)
        learner = MADDPG_Learner(policy, config, device=device)
        buffer = MARL_OffPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 10000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            actions = self.policy.sample_actions(obs_tensors, epsilon=epsilon)
        result = {}
        for k, v in actions.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 0 or v.size == 1:
                    result[k] = v.item()
                else:
                    result[k] = v.squeeze(0)
            else:
                result[k] = v
        return result


class IDDPG_Agents(OffPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        policy = Independent_DDPG_Policy(env.observation_space, env.action_space, config,
                                          self.n_agents, self.agent_keys, device)
        learner = IDDPG_Learner(policy, config, device=device)
        buffer = MARL_OffPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 10000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            actions = self.policy.sample_actions(obs_tensors, epsilon=epsilon)
        result = {}
        for k, v in actions.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 0 or v.size == 1:
                    result[k] = v.item()
                else:
                    result[k] = v.squeeze(0)
            else:
                result[k] = v
        return result


class IPPO_Agents(OnPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        policy = Categorical_MAAC_Policy(env.observation_space, env.action_space, config,
                                          self.n_agents, self.agent_keys, device=device)
        learner = IPPO_Learner(policy, config, device=device)
        buffer = MARL_OnPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 2000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            logits = self.policy(obs_tensors)
            values = self.policy.get_value(obs_tensors)
        actions = self.policy.sample_actions(logits, epsilon=epsilon)
        actions_scalar = {k: v.item() if isinstance(v, np.ndarray) else v for k, v in actions.items()}
        return actions_scalar, logits, values

    def store(self, step_data):
        self.buffer.store(step_data)


class MAPPO_Agents(OnPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        policy = Categorical_MAAC_Policy(env.observation_space, env.action_space, config,
                                          self.n_agents, self.agent_keys, device=device)
        # MAPPO: replace critics with state-based critics
        for key in self.agent_keys:
            policy.critics[key] = nn.Sequential(
                nn.Linear(self.state_dim, 64, device=device),
                nn.ReLU(),
                nn.Linear(64, 1, device=device),
            )
        learner = MAPPO_Learner(policy, config, device=device)
        buffer = MARL_OnPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 2000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            logits = self.policy(obs_tensors)
            # For MAPPO, use state-based values
            state = np.concatenate(list(obs.values()), axis=-1).astype(np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            values = {k: self.policy.critics[k](state_t) for k in self.agent_keys}
        actions = self.policy.sample_actions(logits, epsilon=epsilon)
        actions_scalar = {k: v.item() if isinstance(v, np.ndarray) else v for k, v in actions.items()}
        return actions_scalar, logits, values

    def store(self, step_data):
        self.buffer.store(step_data)


class COMA_Agents(OnPolicyMARLAgents):
    def __init__(self, config: CArgs, env, device=None):
        self.config = config
        self.device = device
        self.n_agents = env.n_agents
        self.agent_keys = env.agent_keys
        self.act_dim = space2shape(env.action_space)
        self.obs_dim = space2shape(env.observation_space)
        self.state_dim = space2shape(env.state_space)

        policy = COMA_Policy(env.observation_space, env.action_space, config,
                              self.n_agents, self.agent_keys, device)
        learner = COMA_Learner(policy, config, device=device)
        buffer = MARL_OnPolicyBuffer(
            config.buffer_size if hasattr(config, 'buffer_size') else 2000,
            self.agent_keys, self.obs_dim, self.act_dim, self.state_dim,
            self.n_agents, device)
        super().__init__(policy, learner, buffer, config, device)

    def act(self, obs, epsilon=0.0):
        obs_tensors = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            logits = self.policy(obs_tensors)
        actions = self.policy.sample_actions(logits, epsilon=epsilon)
        return {k: v.item() if isinstance(v, np.ndarray) else v for k, v in actions.items()}

    def store(self, step_data):
        self.buffer.store(step_data)


# Agent registry
AGENT_REGISTRY = {
    'iql': IQL_Agents,
    'vdn': VDN_Agents,
    'qmix': QMIX_Agents,
    'wqmix': WQMIX_Agents,
    'maddpg': MADDPG_Agents,
    'iddpg': IDDPG_Agents,
    'ippo': IPPO_Agents,
    'mappo': MAPPO_Agents,
    'coma': COMA_Agents,
}

# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

class Runner:
    """Training/evaluation runner for MARL algorithms."""
    def __init__(self, agent, env, config: CArgs):
        self.agent = agent
        self.env = env
        self.config = config
        self.device = config.device
        self.training_steps = 0
        self.episode_rewards = []

    def run(self, mode: str = 'train'):
        if mode == 'train':
            return self._train()
        else:
            return self._evaluate()

    def _train(self):
        n_steps = self.config.training_steps if hasattr(self.config, 'training_steps') else 20000
        log_interval = self.config.log_interval if hasattr(self.config, 'log_interval') else 1000
        eval_interval = self.config.eval_interval if hasattr(self.config, 'eval_interval') else 5000
        epsilon = self.config.epsilon_start if hasattr(self.config, 'epsilon_start') else 1.0
        epsilon_end = self.config.epsilon_end if hasattr(self.config, 'epsilon_end') else 0.01
        epsilon_decay = self.config.epsilon_decay if hasattr(self.config, 'epsilon_decay') else 0.995

        episode = 0
        obs, state, infos = self.env.reset()
        episode_reward = 0.0
        episode_step = 0
        max_episode_steps = self.env.max_episode_steps

        for step in range(n_steps):
            self.training_steps = step

            # Act
            self.agent.train()
            result = self.agent.act(obs, epsilon=epsilon)
            if isinstance(result, tuple):
                actions, logits, values = result
                # Store log_probs and values for on-policy
                log_probs = {}
                for k in self.env.agent_keys:
                    l = logits[k]
                    dist = torch.distributions.Categorical(logits=l)
                    log_probs[k] = dist.log_prob(
                        torch.tensor(actions[k], device=self.device)).cpu().numpy()
                values_np = {k: v.squeeze(0).cpu().numpy() for k, v in values.items()}
            else:
                actions = result
                log_probs = None
                values_np = None

            # Step environment
            obs_next, rewards, dones, terms, truncs, state_next, infos = self.env.step(actions)

            # Store transition
            terminals = {k: terms.get(k, False) for k in self.env.agent_keys}
            step_data = {
                'obs': obs,
                'obs_next': obs_next,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'terminals': terminals,
                'state': state,
                'state_next': state_next,
            }
            if log_probs is not None:
                step_data['log_probs'] = log_probs
            if values_np is not None:
                step_data['values'] = values_np
            self.agent.store(step_data)

            # Update
            if len(self.agent.buffer) >= (self.agent.batch_size if hasattr(self.agent, 'batch_size') else 32):
                info = self.agent.update()

            # Track rewards
            for k in self.env.agent_keys:
                episode_reward += rewards.get(k, 0)
            episode_step += 1

            obs = obs_next
            state = state_next

            # Episode end or reset
            if any(dones.values()) or episode_step >= max_episode_steps:
                self.episode_rewards.append(episode_reward)
                episode += 1
                if step % log_interval == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    print(f"[Step {step}/{n_steps}] Ep {episode}: avg reward = {avg_reward:.3f}, "
                          f"epsilon = {epsilon:.3f}")
                obs, state, infos = self.env.reset()
                episode_reward = 0.0
                episode_step = 0

            # Epsilon decay
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Training complete after {n_steps} steps.")
        return {'episode_rewards': self.episode_rewards}

    def _evaluate(self):
        n_episodes = self.config.eval_episodes if hasattr(self.config, 'eval_episodes') else 10
        self.agent.eval()
        total_rewards = []

        for ep in range(n_episodes):
            obs, state, infos = self.env.reset()
            episode_reward = 0.0
            step = 0
            while step < self.env.max_episode_steps:
                result = self.agent.act(obs, epsilon=0.0)
                if isinstance(result, tuple):
                    actions = result[0]
                else:
                    actions = result
                obs, rewards, dones, terms, truncs, state, infos = self.env.step(actions)
                for k in self.env.agent_keys:
                    episode_reward += rewards.get(k, 0)
                step += 1
                if any(dones.values()):
                    break
            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Evaluation over {n_episodes} episodes: avg reward = {avg_reward:.3f} +/- {std_reward:.3f}")
        return {'avg_reward': avg_reward, 'std_reward': std_reward, 'rewards': total_rewards}


# ──────────────────────────────────────────────
# get_runner() — Main entry point
# ──────────────────────────────────────────────

def get_runner(algo: str, env: str, env_id: str, parser_args=None):
    """
    Create and return a runner for the given algorithm and environment.
    Mimics xuance.get_runner().

    Args:
        algo: Algorithm name (iql, vdn, qmix, wqmix, maddpg, iddpg, ippo, mappo, coma)
        env: Environment backend (mpe, robotic_warehouse)
        env_id: Environment ID (e.g., simple_spread_v3, rware-tiny-2ag-v2)
        parser_args: Optional dict or Namespace of config overrides
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert Namespace to dict
    if isinstance(parser_args, Namespace):
        parser_args = vars(parser_args)
    parser_args = parser_args or {}

    # Default config per algorithm
    defaults = {
        # Off-policy params
        'learning_rate': 1e-3,
        'critic_learning_rate': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'sync_frequency': 100,
        'buffer_size': 20000,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        # On-policy params
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        'epochs': 10,
        # Training
        'training_steps': 20000,
        'log_interval': 1000,
        'eval_interval': 5000,
        'eval_episodes': 10,
        # Network
        'representation': 'Basic_MLP',
        'representation_hidden': [64, 64],
        'q_hidden_sizes': [64, 64],
        # WQMIX
        'alpha': 0.5,
    }

    # Merge defaults with parser_args (parser_args overrides)
    config = CArgs(**defaults)
    for k, v in parser_args.items():
        if k == 'running_steps':
            config.training_steps = v
        else:
            setattr(config, k, v)

    config.device = device

    # Create environment
    env_instance = DummyVecMultiAgentEnv(lambda: _get_env(env_id, algo))
    config.state_space = env_instance.state_space

    # Override max_episode_steps if provided
    if hasattr(config, 'max_episode_steps') and config.max_episode_steps is not None:
        env_instance.max_episode_steps = config.max_episode_steps

    # Select algorithm
    algo_lower = algo.lower().replace('_', '')
    if algo_lower not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algo}. "
                         f"Available: {list(AGENT_REGISTRY.keys())}")

    agent_class = AGENT_REGISTRY[algo_lower]
    agent = agent_class(config, env_instance, device)

    # Create runner
    runner = Runner(agent, env_instance, config)
    return runner

# ──────────────────────────────────────────────
# rware-tiny-2ag-v2 dedicated training & evaluation
# ──────────────────────────────────────────────
# rware-tiny-2ag-v2 specifics:
#   - 2 agents, Discrete(5) action space (NOOP/FORWARD/LEFT/RIGHT/TOGGLE_LOAD)
#   - observation: Box(71,) per agent (flattened: 8 self-features + 63 sensor features)
#   - reward: INDIVIDUAL — only the agent that delivers a requested shelf gets +1
#   - state (for QMIX/VDN mixing + MAPPO critic): global concat of all observations (142,)
#   - max_episode_steps: 500, request_queue_size: 2
#   - delivery is the only reward event; no penalty for time
# ──────────────────────────────────────────────

RWARE_ENV_IDS = {'rware-tiny-2ag-v2', 'rware-tiny-4ag-v2', 'rware-small-4ag-v2',
                 'rware-small-6ag-v2', 'rware-medium-8ag-v2'}


def train_rware(algo: str, env_id: str = 'rware-tiny-2ag-v2',
                training_steps: int = 200_000, batch_size: int = 32,
                buffer_size: int = 50000, render: bool = False,
                eval_interval: int = 10_000, eval_episodes: int = 10,
                device=None):
    """Dedicated training loop for rware environments with periodic evaluation.

    Returns a dict with 'train_rewards' (per-episode returns during training)
    and 'eval_results' (list of dicts with avg_reward at each eval point).
    """
    import warnings
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_dict = {
        'training_steps': training_steps,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'render': render,
        'render_mode': 'human' if render else None,
        'eval_interval': eval_interval,
        'eval_episodes': eval_episodes,
        'log_interval': max(1000, training_steps // 20),
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995,
        'max_episode_steps': 500,
    }

    runner = get_runner(algo, 'robotic_warehouse', env_id, config_dict)
    runner.agent.device = device
    runner.device = device

    eval_results = []

    if hasattr(runner.config, 'eval_interval') and runner.config.eval_interval > 0:
        original_step = runner.agent.act
        eval_counter = [0]

        def step_and_eval(obs, epsilon=0.0):
            result = original_step(obs, epsilon)
            eval_counter[0] += 1
            if eval_counter[0] % runner.config.eval_interval == 0:
                print(f'\n[Eval at step {eval_counter[0]}]')
                eval_out = evaluate_rware(runner, n_episodes=runner.config.eval_episodes,
                                          render=False)
                eval_results.append({'step': eval_counter[0], **eval_out})
            return result

        if isinstance(runner.agent, OffPolicyMARLAgents) or isinstance(runner.agent, OnPolicyMARLAgents):
            runner.agent.act = step_and_eval

    train_out = runner.run(mode='train')
    return {
        'train_rewards': train_out.get('episode_rewards', []),
        'eval_results': eval_results,
    }


def evaluate_rware(runner, n_episodes: int = 10, render: bool = False):
    """Evaluate on rware, tracking rware-specific metrics.

    Returns a dict with:
      - avg_reward: mean per-episode return (sum across agents)
      - std_reward: standard deviation
      - avg_deliveries: mean number of successful deliveries per episode
      - avg_episode_length: mean steps per episode
    """
    runner.agent.eval()
    total_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, state, infos = runner.env.reset()
        episode_reward = 0.0
        step = 0

        while step < runner.env.max_episode_steps:
            result = runner.agent.act(obs, epsilon=0.0)
            if isinstance(result, tuple):
                actions = result[0]
            else:
                actions = result

            obs, rewards, dones, terms, truncs, state, infos = runner.env.step(actions)

            for k in runner.env.agent_keys:
                episode_reward += rewards.get(k, 0)
            step += 1

            if any(dones.values()):
                break

        total_rewards.append(episode_reward)
        episode_lengths.append(step)

    runner.agent.train()

    avg_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    avg_len = float(np.mean(episode_lengths))
    print(f"  eval: avg_reward={avg_reward:.3f} ± {std_reward:.3f}, "
          f"avg_ep_len={avg_len:.1f}")

    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_episode_length': avg_len,
        'rewards': total_rewards,
    }


# ──────────────────────────────────────────────
# Main block (compatible with test_xuance.py)
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='iql')
    parser.add_argument('--env', type=str, default='mpe')
    parser.add_argument('--env_id', type=str, default='simple_spread_v3')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--training_steps', type=int, default=20000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--buffer_size', type=int, default=20000)
    args = parser.parse_args()

    runner = get_runner(
        algo=args.algo,
        env=args.env,
        env_id=args.env_id,
        parser_args=vars(args),
    )
    runner.run(mode=args.mode)
