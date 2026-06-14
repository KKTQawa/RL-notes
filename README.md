# RL-notes

强化学习算法实现笔记，包含多种经典RL算法在不同环境中的实现。

每个算法-环境组合都有独立的脚本，便于理解算法流程和实验对比。

## 算法列表

### 基于价值的方法
- **Q-Learning**: Q_table-MountainCar_v0.py, Q_table-Acrobot-v1.py
- **DQN**: DQN-CartPole.py
- **Value Iteration**: value_iteration-FrozenLake.py

### 基于策略的方法
- **REINFORCE**: REINFORCE-InvertedPendulum-v4.py

### Actor-Critic方法
- **A2C**: A2C-cartpole.py
- **PPO**: PPO-Pendulum.py

### 确定性策略梯度方法
- **DDPG**: DDPG-Pendulum.py
- **TD3**: TD3-Pendulum.py

### 最大熵方法
- **SAC**: SAC-HalfCheetah.py, SAC-Pendulum_actor.pt, SAC-BC-Pendulum.py

## 环境列表

- CartPole-v1
- Pendulum-v1
- MountainCar-v0
- Acrobot-v1
- InvertedPendulum-v4
- HalfCheetah-v4
- FrozenLake-v1

## 项目结构

```
.
├── README.md
├── all.txt                    # 包含所有gymnasium环境的id
├── test_env.py                # 用于快速查看环境信息
├── model/                     # 保存训练好的模型
├── saves/                     # 保存训练过程数据
├── *.py                       # 各算法实现脚本
├── *.md                       # 算法说明文档
├── *_train.png                # 训练曲线图
└── *_actor_net.pt             # 训练好的模型权重
```

## 安装

**环境要求**: Python 3.10+

```bash
pip install gymnasium ale-py numpy torch matplotlib
```

## 使用方法

每个算法脚本都可以独立运行：

```bash
# 例如运行DQN算法
python DQN-CartPole.py

# 查看环境信息
python test_env.py
```

## 特性

- 每个算法都有完整的实现和训练可视化
- 包含训练曲线图和保存的模型权重
- 代码结构清晰，便于学习和修改
- 支持多种经典强化学习环境

## 依赖项

- gymnasium: 强化学习环境接口
- ale-py: Atari学习环境
- numpy: 数值计算
- torch: 深度学习框架
- matplotlib: 数据可视化

