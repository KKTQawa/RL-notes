# VDN (Value Decomposition Networks)

## 概述

VDN (Value Decomposition Networks) 是一种用于**合作式多智能体强化学习**的 value-based 算法，由 Sunehag et al. (2017) 提出。核心思想是：将**联合动作值函数 $Q_{tot}$ 分解为各智能体个体动作值函数 $Q_i$ 的加和**，从而在 CTDE (Centralized Training with Decentralized Execution) 框架下实现有效的协作学习。

$$
Q_{tot}(\boldsymbol{\tau}, \boldsymbol{a}) = \sum_{i=1}^{n} Q_i(\tau_i, a_i)
$$

VDN 通过简单的加性分解保证了 IGM (Individual-Global-Max) 性质：对 $Q_{tot}$ 做全局 argmax 等价于每个 $Q_i$ 分别做 argmax，从而允许分布式执行时各智能体仅根据自身 $Q_i$ 贪心选择动作。

---

## 1. 网络结构

所有智能体共享同一套 Q 网络参数，用 one-hot分配agent ID

Q_agenti->Net(representation特征提取(BasicMLP)->BasicQhead)->VDNMixer(合并Q_0,Q_1,...Q_n)->Q_tot

### 1.1 BasicMLP（表示/特征提取层representation）

将观测映射为隐藏状态向量，输入为原始观测拼接 one-hot agent ID：

$$
\text{input}_i = [o_i, \text{onehot}(i)]
$$

设隐藏层维度序列为 $H = [h^{(1)}, \dots, h^{(L)}]$，则：

$$
h^{(0)} = \text{input}_i,\qquad
h^{(k)} = \text{ReLU}(W^{(k)} h^{(k-1)} + b^{(k)}),\ k=1,\dots,L
$$

输出为最后一层隐藏状态：

$$
\text{state}_i = h^{(L)} \in \mathbb{R}^{h^{(L)}}
$$

默认配置 $H = [128, 128]$（两个隐藏层各 128 维）。

### 1.2 BasicQhead（输出头）

将隐藏状态映射为各动作的 Q 值：

$$
Q_i(o_i, \cdot) = W^{(M+1)} \cdot \text{ReLU}(\dots \text{ReLU}(W^{(1)} \cdot \text{state}_i + b^{(1)}) \dots) + b^{(M+1)}
$$

输出维度为动作空间大小。

### 1.3 VDNMixer

直接求和：

$$
Q_{tot} = \sum_{i=1}^{n} Q_i
$$

无需额外的网络参数，计算开销极小。

### 1.4 概括

每个智能体的输入->特征提取器->输出头->合并器->联合输出

---

## 2. IGM 性质

VDN 的加性分解天然满足 IGM (Individual-Global-Max) 性质：

$$
\underset{\mathbf{a}}{\arg\max}\; Q_{tot}(\boldsymbol{\tau}, \mathbf{a}) =
\begin{pmatrix}
\underset{a_1}{\arg\max}\; Q_1(\tau_1, a_1) \\
\vdots \\
\underset{a_n}{\arg\max}\; Q_n(\tau_n, a_n)
\end{pmatrix}
$$

**证明**：由于 $Q_{tot} = \sum_i Q_i$，且各 $Q_i$ 的解耦（每个 $Q_i$ 仅依赖 $\tau_i$ 和 $a_i$），对 $Q_{tot}$ 的全局 argmax 等价于对各 $Q_i$ 分别 argmax：

$$
\underset{\mathbf{a}}{\arg\max}\; \sum_{i=1}^{n} Q_i(\tau_i, a_i) =
\big( \underset{a_1}{\arg\max}\; Q_1(\tau_1, a_1),\ \dots,\ \underset{a_n}{\arg\max}\; Q_n(\tau_n, a_n) \big)
$$

这使得在执行阶段，每个智能体只需根据自己当前的 $Q_i$ 贪心选择动作，无需知道其他智能体的信息或状态。

---

## 3. 训练流程

### Step 1: 环境交互（Data Collection）

每个 step 将所有智能体的观测拼接 batch，使用 $\epsilon$-greedy 策略选择动作：

$$
a_i^t = \begin{cases}
\arg\max_{a} Q_i(o_i^t, a) & \text{概率 } 1-\epsilon \\
\text{随机动作} & \text{概率 } \epsilon
\end{cases}
$$

执行联合动作 $\mathbf{a}^t = (a_1^t, \dots, a_n^t)$，从环境获得下一个观测 $\mathbf{o}^{t+1}$ 和奖励 $\mathbf{r}^{t+1}$。将 $(\mathbf{o}^t, \mathbf{a}^t, \mathbf{r}^{t+1}, \mathbf{o}^{t+1}, done)$ 存入 Replay Buffer。

### Step 2: 从 Replay Buffer 采样

当 buffer 中样本数超过 `start_training` 后，每步从 buffer 中随机采样一个 batch。

### Step 3: 计算 TD 目标

**基本版本：**

$$
Q_{tot}^{target} = \bar{R} + \gamma (1 - done) \cdot \max_{\mathbf{a}'} Q_{tot}^{target\_net}(\mathbf{o}^{t+1}, \mathbf{a}')
$$

其中 $\bar{R} = \frac{1}{n} \sum_{i=1}^{n} r_i$ 为各智能体奖励的均值。

**Double Q 版本：**

使用当前网络选动作，目标网络算值，减小过估计：

$$
\mathbf{a}^{*, t+1} = \arg\max_{\mathbf{a}'} Q_{tot}^{eval}(\mathbf{o}^{t+1}, \mathbf{a}')
$$

$$
Q_{tot}^{target} = \bar{R} + \gamma (1 - done) \cdot Q_{tot}^{target\_net}(\mathbf{o}^{t+1}, \mathbf{a}^{*, t+1})
$$

### Step 4: 计算 Loss

使用 MSE Loss ：

$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathcal{D}}\left[\left(Q_{tot}^{eval} - Q_{tot}^{target}\right)^2\right]
$$

其中 $Q_{tot}^{eval} = \sum_i Q_i(o_i^t, a_i^t)$，$Q_{tot}^{target}$ 由上一步计算得到（detach 阻止梯度回传）。

### Step 5: 梯度更新

反向传播计算梯度，可选先梯度裁剪后更新网络参数：

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

### Step 6: 周期性同步 Target Network

每隔 `sync_frequency` 次训练迭代，将当前网络的参数完整复制到目标网络：

$$
\theta^- \leftarrow \theta
$$

### Step 7: $\epsilon$ 更新

每个 step 后按线性衰减更新 $\epsilon$：

$$
\epsilon \gets \max(\epsilon_{end},\; \epsilon - \Delta_\epsilon),
\qquad
\Delta_\epsilon = \frac{\epsilon_{start} - \epsilon_{end}}{decay\_step\_greedy}
$$

经过 `decay_step_greedy` 步，最终 $\epsilon$ 会从 `start_greedy` 衰减到 `end_greedy`。

---

## 4. VDN vs QMIX

| 对比维度 | VDN | QMIX |
|---------|-----|------|
| 分解方式 | 加性分解 $Q_{tot} = \sum Q_i$ | 单调非线性分解 $Q_{tot} = f(Q_1, \dots, Q_n)$ |
| 混合网络 | 无参数，直接求和 | 超网络（Hypernetwork），需额外参数 |
| 表达能力 | 弱（假设智能体独立） | 强（可建模智能体交互） |
| IGM 保证 | 天然满足 | 通过单调性约束保证 |
| 计算开销 | 极低 | 较低 |
| 适用场景 | 简单协作任务 | 复杂协作任务 |

---

## 5. 关键超参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `env_id` | 环境 ID | `rware-tiny-4ag-v2` |
| `device` | 计算设备 | `cuda` 或 `cpu` |
| `representation_hidden_size` | MLP 表示层隐藏层维度 | [128, 128] |
| `q_hidden_size` | Q Head 隐藏层维度 | [128] |
| `buffer_size` | Replay Buffer 容量 | 50000 |
| `batch_size` | 采样 batch 大小 | 256 |
| `learning_rate` | 学习率 | 0.0005 |
| `gamma` | 折扣因子 | 0.99 |
| `double_q` | 是否使用 Double Q-learning | True |
| `start_greedy` | 初始 $\epsilon$ | 1.0 |
| `end_greedy` | 最终 $\epsilon$ | 0.05 |
| `decay_step_greedy` | $\epsilon$ 衰减步数 | 200000 |
| `start_training` | 开始训练前的步数 | 1000 |
| `running_steps` | 总运行步数 | 10000000 |
| `sync_frequency` | 目标网络同步间隔（次训练） | 200 |
| `use_grad_clip` | 是否使用梯度裁剪 | True |
| `grad_clip_norm` | 梯度裁剪范数 | 10.0 |

---

## 6. 环境信息

- **任务**：`rware-tiny-4ag-v2`（Robotic Warehouse，4 个智能体）
- **动作空间**：5 个离散动作（前进、左转、右转、停留、装载）
- **观测维度**：71
- **状态空间**：各智能体观测拼接（`obs_dim × n_agents = 284`）

