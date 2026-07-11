# IQL (Independent Q-Learning) 算法

## 概述

IQL (Independent Q-Learning) 是一种多智能体强化学习算法，本质上是将单智能体 DQN 直接扩展到多智能体场景。每个智能体独立学习自己的 Q 函数，将其他智能体视为环境的一部分。尽管每个智能体独立决策，但实验表明 IQL 在许多合作任务中仍能取得不错的效果。

本实现基于 Xuance 库的 IQL 实现，在不 import xuance 的前提下重新实现了核心算法，并在 `rware-tiny-4ag-v2` (Robotic Warehouse, 4个智能体) 环境下进行训练和推理。

---

## 算法架构

### 核心思想

每个智能体 `i` 维护一个独立的 Q 函数 $Q_i(s_i, a_i)$，其中 $s_i$ 是智能体 i 的局部观测，$a_i$ 是其动作。所有智能体独立与环境交互，使用标准的 Q-learning 更新规则：

$$Q_i(s_i, a_i) \leftarrow Q_i(s_i, a_i) + \alpha \left[ r_i + \gamma \max_{a'_i} Q_i(s'_i, a'_i) - Q_i(s_i, a_i) \right]$$

### 关键技术

| 技术 | 说明 |
|------|------|
| **经验回放 (Replay Buffer)** | 存储历史经验 $(s, a, r, s')$，打破数据相关性 |
| **目标网络 (Target Network)** | 使用独立的目标网络计算 TD target，提高训练稳定性，每 `sync_frequency` 步同步一次 |
| **Double Q-learning** | 用 eval 网络选择动作，target 网络评估价值，减少 Q 值过估计 |
| **Epsilon-Greedy 探索** | 以 $\epsilon$ 概率随机探索，$\epsilon$ 从 `start_greedy` 线性衰减到 `end_greedy` |
| **参数共享 (Parameter Sharing)** | 所有智能体共享同一套网络参数，通过 one-hot agent ID 区分不同智能体 |

### 网络结构

```
每个智能体观测 (obs_dim=71)
    │
    ▼
┌─────────────────────────────┐
│  Basic_MLP (表示网络)        │
│  Linear(71, 64) + ReLU       │
│  输出: 64维特征向量           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  拼接 Agent ID (one-hot, 4维)│
│  输入维度: 64 + 4 = 68       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  BasicQhead (Q网络头部)      │
│  Linear(68, 64) + ReLU      │
│  Linear(64, 5)  (动作数)     │
│  输出: 各动作的 Q 值          │
└─────────────────────────────┘
```

- **Eval 网络**: 用于选择动作和计算当前 Q 值
- **Target 网络**: Eval 网络的深拷贝，用于计算 TD target，定期同步

---

## 训练过程

### 数据流

```
1. 初始化 N 个并行环境 (parallels=N)
2. 每个训练步:
   a. 从 N 个环境获取观测 obs_dict (list of dicts)
   b. 展平所有智能体的观测: [N, n_agents * obs_dim]
   c. 创建 agent IDs (one-hot): [N * n_agents, n_agents]
   d. Eval 网络前向: -> argmax actions
   e. Epsilon-greedy 探索: 以 epsilon 概率替换为随机动作
   f. 执行动作，获取 next_obs, reward, done
   g. 存储经验到 Replay Buffer
   h. 当 current_step >= start_training:
      - 从 buffer 采样 batch_size 条经验
      - 计算 TD loss，更新 eval 网络
      - 每 sync_frequency 步同步 target 网络
   i. 更新 epsilon: epsilon -= delta_egreedy
   j. 处理 episode 终止（自动 reset 环境）
```

### Loss 函数

使用均方误差 (MSE) 作为损失函数：

$$L = \frac{1}{\sum mask} \sum (Q_{eval}(s,a) - Q_{target})^2 \cdot mask$$

其中 TD target 为：

$$Q_{target} = r + \gamma \cdot (1 - done) \cdot Q_{next}$$

使用 Double Q-learning 时：

$$a^* = \arg\max_{a'} Q_{eval}(s', a')$$
$$Q_{next} = Q_{target}(s', a^*)$$

### 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `gamma` | 0.99 | 折扣因子 |
| `learning_rate` | 0.001 | Adam 优化器学习率 |
| `batch_size` | 256 | 采样批次大小 |
| `buffer_size` | 50000 | 经验池容量 |
| `start_greedy` | 1.0 | 初始探索率 |
| `end_greedy` | 0.05 | 最终探索率 |
| `decay_step_greedy` | 500000 | 探索率衰减步数 |
| `sync_frequency` | 200 | 目标网络同步频率 |
| `start_training` | 1000 | 开始训练前的预热步数 |
| `double_q` | True | 是否使用 Double Q-learning |

---

## 推理过程

### 数据流

```
1. 加载训练好的模型权重
2. 创建环境实例
3. 对每个 episode:
   a. 重置环境，获取初始观测
   b. 对每个时间步:
      - 展平观测并添加 agent ID
      - Eval 网络前向 -> argmax actions (epsilon=0, 纯贪婪)
      - 执行动作，获取 next_obs, reward
      - 累加 episode 奖励
      - 判断是否终止
   c. 输出 episode 总奖励和步数
4. 输出所有 episode 的平均奖励
```

推理时 `test_mode=True`，不使用 epsilon-greedy 探索，直接选择最大 Q 值对应的动作。

---

## 代码结构 (iql-rwarehouse.py)

| 组件 | 行号 | 说明 |
|------|------|------|
| `IQLConfig` | 40 | 配置类 |
| `mlp_block()` | 72 | MLP 层构建函数 |
| `Basic_MLP` | 80 | 表示网络 |
| `BasicQhead` | 96 | Q 网络头部 |
| `BasicQnetwork` | 112 | 组合网络 (eval + target) |
| `ReplayBuffer` | 163 | 经验回放缓冲区 |
| `ParallelEnv` | 224 | 并行环境管理器 |
| `IQLAgent` | 274 | IQL 智能体 (训练 + 推理) |
| `main()` | 737 | 入口函数 |
