# RL-notes

强化学习算法实现笔记，包含多种经典单智能体与多智能体 RL 算法在不同环境中的实现。

每个算法-环境组合都有独立的脚本，便于理解算法流程和实验对比。

## 算法与环境

### 单智能体 (Single-Agent RL)

| 算法 | 环境 | 文件 |
|------|------|------|
| Value Iteration | FrozenLake-v1 | `value_iteration-FrozenLake.py` |
| Q-Learning (Tabular) | MountainCar-v0 | `Q_table-MountainCar_v0.py` |
| Q-Learning (Tabular) | Acrobot-v1 | `Q_table-Acrobot-v1.py` |
| REINFORCE | InvertedPendulum-v4 | `REINFORCE-InvertedPendulum-v4.py` |
| DQN + Reward Shaping | CartPole-v0 | `DQN-CartPole.py` |
| A2C | CartPole-v1 | `A2C-cartpole.py` |
| DDPG | Pendulum-v1 | `DDPG-Pendulum.py` |
| TD3 | Pendulum-v1 | `TD3-Pendulum.py` |
| PPO | Pendulum-v1 | `PPO-Pendulum.py` |
| SAC | Pendulum-v1 | `SAC-BC-Pendulum.py` |
| SAC | HalfCheetah-v5 | `SAC-HalfCheetah.py` |

### 多智能体 (Multi-Agent RL)

#### PettingZoo

| 算法 | 环境 | 文件 |
|------|------|------|
| VDN | Pistonball-v6 | `multiRL/pettingzoo_code/VDN-fali.py` |
| IQL | Go-v5 | `multiRL/pettingzoo_code/IQL_go_v5-fail.py` |

#### RWARE (Robotic Warehouse)

| 算法 | 环境 | 文件 |
|------|------|------|
| VDN | rware-tiny-2ag-v2 | `multiRL/rware_code/vdn-rwarehouse.py` |
| QMIX | rware-tiny-4ag-v2 | `multiRL/rware_code/qmix-rwarehouse.py` |
| IQL | rware-tiny-4ag-v2 | `multiRL/rware_code/iql-rwarehouse-test.py` |
| MAPPO | rware-tiny-4ag-v2 | `multiRL/rware_code/mappo-rwarehouse.py` |
| Human Play | rware-tiny | `multiRL/rware_code/rware_human_play.py` |

#### ViZDoom

| 算法 | 环境 | 文件 |
|------|------|------|
| DQN (CNN) | VizdoomBasic-v1 | `multiRL/vizdoom_code/DQN-VizdoomBasic-v1.py` |
| DQN (CNN) | 多环境 (HealthGathering, DeadlyCorridor, TakeCover 等) | `multiRL/vizdoom_code/DQN-Vizdoomx.py` |

#### Xuance (统一框架)

| 算法 | 环境 | 文件 |
|------|------|------|
| IQL / VDN / QMIX / MAPPO / MADDPG / IPPO / WQMIX | MPE + RWARE | `multiRL/xuance_code/algorithm.py` |

#### MetaDrive

| 用途 | 文件 |
|------|------|
| 驾驶仿真环境测试 | `multiRL/metadrive_code/test_metadrive.py` |

## 项目结构

```
.
├── *.py                       # 单智能体算法实现脚本
├── *.md                       # 对应算法说明文档
├── *_train.png                # 训练曲线图
├── *_actor_net.pt             # 训练好的动作网络权重
├── all.txt                    # 所有 gymnasium 环境 ID 列表
├── test_env.py                # 环境信息查看工具
│
├── multiRL/                   # 多智能体强化学习
│   ├── pettingzoo_code/       # PettingZoo 环境 (Pistonball, Go)
│   ├── rware_code/            # RWARE 仓库机器人环境 (VDN, QMIX, IQL, MAPPO)
│   │   └── models/            # 训练好的模型权重
│   ├── vizdoom_code/          # ViZDoom FPS 环境 (DQN + CNN)
│   │   └── image/             # 训练曲线截图
│   ├── metadrive_code/        # MetaDrive 驾驶仿真
│   └── xuance_code/           # Xuance 框架实现与自实现 MARL 算法
│       ├── algorithm.py       # 自实现 IQL/VDN/QMIX/MAPPO/MADDPG/IPPO/WQMIX
│       ├── test_xuance.py     # Xuance 库调用示例
│       ├── models/            # 训练好的模型权重
│       └── logs/              # TensorBoard 日志
│
├── saves/                     # 训练过程数据
└── README.md
```

## 安装

**环境要求**: Python 3.10.x左右，3.10.20亲测通过

```bash
pip install gymnasium ale-py numpy matplotlib
```

### 可选gymnasium模块

```bash
# Atari 环境
pip install "gymnasium[atari]"

# Box2D 环境
pip install "gymnasium[box2d]"

# MuJoCo 环境 (HalfCheetah 等)
pip install "gymnasium[mujoco]"

# 全部 gymnasium 环境
pip install "gymnasium[all]"
```

### 多智能体依赖

实测python 3.10.20环境可以包揽下面的所有环境：
```cmd
# PettingZoo
pip install pettingzoo

# RWARE
pip install git+https://github.com/semitable/robotic-warehouse.git

# ViZDoom
pip install git+https://github.com/Farama-Foundation/ViZDoom.git

# MetaDrive
pip install git+https://github.com/metadriverse/metadrive.git

# Xuance
pip install git+https://github.com/agi-brain/xuance.git
```
请尽量从github仓库安装，以确保获得最新的版本;使用pip 安装的有遗留bug


## 说明文档

每个算法在对应 `.md` 文件中有详细说明，覆盖算法原理、网络结构、损失函数、训练曲线等。多智能体部分另有 `rware_START.md` 环境入门指南和详细算法笔记。
