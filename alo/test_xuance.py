import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import OUT
from types import SimpleNamespace as SN
import yaml
import gymnasium as gym
import rware

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'rware-tiny-2ag-v2.yaml')
    basic_path = os.path.join(os.path.dirname(__file__), 'basic.yaml')

    algo_config = OUT.load_yaml(config_path)
    basic_config = OUT.load_yaml(basic_path)
    full_config = OUT.deepcopy(basic_config)
    full_config.update(algo_config)
    config = SN(**full_config)

    config.device = 'cpu'
    config.render_mode = 'rgb_array'
    config.use_tqdm = True

    temp_env = gym.make(config.env_id)
    config.n_agents = len(temp_env.action_space)
    config.obs_dim = temp_env.observation_space[0].shape[0]
    config.act_dim = temp_env.action_space[0].n
    config.state_dim = config.obs_dim * config.n_agents
    temp_env.close()

    print(f"QMIX on {config.env_id}")
    print(f"  Agents: {config.n_agents}, ObsDim: {config.obs_dim}, ActDim: {config.act_dim}, StateDim: {config.state_dim}")
    print(f"  Device: {config.device}")
    print(f"  Parallels: {config.parallels}")
    print(f"  Running steps: {config.running_steps}")

    # Train
    OUT.train(config)

    # Evaluate
    eval_reward = OUT.evaluate(config)
    print(f"Final evaluation: avg_reward = {eval_reward:.4f}")
