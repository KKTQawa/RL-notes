# QMIX 算法

## 概述

QMIX (Q-MIX) 是一种用于**合作式多智能体强化学习**的 value-based 算法。核心思想是：每个智能体 $i$ 学习独立的 $Q_i(\tau_i, a_i)$（使用 DRQN 处理部分可观测性），然后通过一个**混合网络（Mixing Network）**将所有 $Q_i$ 非线性的组合成全局 $Q_{tot}$，并保证：

$$
\frac{\partial Q_{tot}}{\partial Q_i} \ge 0, \quad \forall i
$$

这个单调性约束确保对 $Q_{tot}$ 做 argmax 等价于每个 $Q_i$ 分别做 argmax，从而允许**集中式训练 + 分布式执行**（CTDE）。

---

## 1. 网络结构

### 1.1 Agent Network（智能体网络）

$$
->MLP->QHead->
$$

$$
Q_i(o_i, a_i) = \text{QHead}(\text{MLP}([o_i \;\|\; \text{one\_hot}(i)]))
$$

### 1.2 Mixing Network（混合网络）

Mixing Network 接收所有智能体的 $Q_i$ 和一个**全局状态 $s_{tot}$**（由所有观测拼接而成 $s_{tot} = [o_1, o_2, \dots, o_n]$），输出 $Q_{tot}$：

$$
Q_{tot} = f_{\text{mix}}(Q_1, Q_2, \dots, Q_n; s_{tot})
$$

内部结构为两层的超网络（Hypernetwork）：

- **第一层权重** $W_1 = |\text{MLP}_1(s_{tot})|$，形状 $(n\_agents \times hidden\_dim)$
- **第一层偏置** $b_1 = \text{Linear}(s_{tot})$  
- **第二层权重** $W_2 = |\text{MLP}_2(s_{tot})|$，形状 $(hidden\_dim \times 1)$
- **第二层偏置** $b_2 = \text{MLP}_3(s_{tot})$

前向计算：

$$
\begin{aligned}
h &= \text{ELU}(Q_{ag} W_1 + b_1) \\
Q_{tot} &= h W_2 + b_2
\end{aligned}
$$

其中 $Q_{ag} \in \mathbb{R}^{1 \times n\_agents}$ 是各智能体的 Q 值向量。

使用 $\text{abs}(\cdot)$ 保证权重为正，从而满足**单调性约束**。

---

## 2. 单调性约束

单调性约束保证了对 $Q_{tot}$ 的全局 argmax 等价于各 $Q_i$ 的局部 argmax 的组合：

$$
\underset{\mathbf{a}}{\arg\max}\; Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) =
\begin{pmatrix}
\underset{a_1}{\arg\max}\; Q_1(\tau_1, a_1) \\
\vdots \\
\underset{a_n}{\arg\max}\; Q_n(\tau_n, a_n)
\end{pmatrix}
$$

这使得在**执行阶段**，每个智能体只需根据自己当前的 $Q_i$ 贪心选择动作，无需知道其他智能体的信息。

---

## 3. 训练流程

### Step 1: 环境交互（data collection）

- 每个 step 将所有智能体的观测拼接 batch，使用 $\epsilon$-greedy 策略选择动作：

$$
a_i^t = \begin{cases}
\arg\max_{a} Q_i(o_i^t, a) & \text{概率 } 1-\epsilon \\
\text{随机动作} & \text{概率 } \epsilon
\end{cases}
$$

- 执行动作，从环境获得下一个观测 $o^{t+1}$ 和奖励 $r^{t+1}$。
- 将  $(o^t, a^t, r^{t+1}, o^{t+1}, s^t, s^{t+1}, done)$ 存入 Replay Buffer。

### Step 2: 从 Replay Buffer 采样

当 buffer 中样本数超过 `start_training` 后，每 `training_frequency` 步从 buffer 中随机采样一个 batch。

### Step 3: 计算 TD 目标

**基本版本（单 Q）：**

$$
Q_{tot}^{target} = \bar{r} + \gamma (1 - done) \cdot Q_{tot}^{target\_net}(o^{t+1}, a^{*, t+1})
$$

其中 $\bar{r} = \frac{1}{N}\sum_i r_i$ 是所有智能体奖励的均值。

**Double Q 版本：**

使用当前网络选动作，目标网络算值：

$$
a^{*, t+1} = \arg\max_{a} Q_i^{eval}(o_i^{t+1}, a)
$$

$$
Q_{tot}^{target} = \bar{r} + \gamma (1 - done) \cdot Q_{tot}^{target\_net}(Q^{target}(o^{t+1}, a^{*, t+1}), s^{t+1})
$$

### Step 4: 计算 Loss

使用 MSE Loss 优化当前网络：

$$
\mathcal{L}(\theta) = \mathbb{E}\left[\left(Q_{tot}^{eval} - Q_{tot}^{target}\right)^2\right]
$$

### Step 5: 梯度更新

反向传播计算梯度，可选先梯度裁剪后更新网络参数

### Step 6: 周期性同步 Target Network

每隔 `sync_frequency` 次训练迭代，将当前网络的参数完整复制到目标网络：

### Step 7: $\epsilon$ 更新

每个 step 后按线性衰减更新 $\epsilon$：

$$
\epsilon \gets \max(\epsilon_{end},\; \epsilon - \Delta_\epsilon)
$$

---

## 4. 关键超参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `env_id` | 环境 ID | `rware-tiny-2ag-v2` |
| `env_seed` | 环境随机种子 | 1 |
| `max_episode_steps` | 单 episode 最大步数 | 500 |
| `device` | 计算设备 | `cpu` |
| `representation_hidden_size` | Agent 编码器隐层维度 | [64] |
| `q_hidden_size` | Q Head 隐层维度 | [128] |
| `hidden_dim_mixing_net` | Mixing Network 隐层维度 | 64 |
| `hidden_dim_hyper_net` | Hypernetwork 隐层维度 | 64 |
| `buffer_size` | Replay Buffer 容量 | 100000 |
| `batch_size` | 采样 batch 大小 | 256 |
| `learning_rate` | 学习率 | 0.001 |
| `gamma` | 折扣因子 | 0.99 |
| `double_q` | 是否使用 Double Q-learning | True |
| `start_greedy` | 初始 $\epsilon$ | 1.0 |
| `end_greedy` | 最终 $\epsilon$ | 0.05 |
| `decay_step_greedy` | $\epsilon$ 衰减步数 | 2500000 |
| `start_training` | 开始训练前的步数 | 1000 |
| `running_steps` | 总运行步数 | 100000 |
| `training_frequency` | 训练间隔（步） | 1 |
| `sync_frequency` | 目标网络同步间隔（次训练） | 100 |
| `use_grad_clip` | 是否使用梯度裁剪 | False |
| `grad_clip_norm` | 梯度裁剪范数 | 0.5 |
| `eval_interval` | 评估间隔（步） | 100000 |
| `test_episode` | 评估时运行的 episode 数 | 1 |
| `model_dir` | 模型保存路径 | `models/qmix/` |
| `mode` | 运行模式（`train`/`test`） | `test` |
| `render_mode` | 渲染模式 | `human` |
