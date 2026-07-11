# MAPPO (Multi-Agent Proximal Policy Optimization)

## 概述

MAPPO 是一种基于 **CTDE (Centralized Training with Decentralized Execution)** 框架的多智能体强化学习算法。它扩展了单智能体 PPO 算法到多智能体场景，核心特点是：

- **Actor**：每个智能体基于自己的局部观测做出决策（分散式执行）
- **Critic**：训练时使用全局状态进行评估（集中式训练）
- **参数共享**：所有智能体共享同一套 Actor-Critic 网络，通过 one-hot agent ID 区分

## 网络架构与细节

所有智能体共享同一套 Actor-Critic 网络，通过 one-hot agent ID 区分不同智能体。

```
 局部观测 o_i (obs_dim)        全局状态 s (state_dim)
  + One-hot ID (n_agents)       + One-hot ID (n_agents)
         |                              |
         ▼                              ▼
   BasicMLP (actor_hidden)        BasicMLP (critic_hidden)
   [Linear+ReLU] × L             [Linear+ReLU] × M
         |                              |
         ▼                              ▼
   ActorNet (Linear 输出层)      CriticNet (Linear 输出层)
         |                              |
         ▼                              ▼
   logits (n_actions)              V(s, i) (1)
```

### Actor 网络 (接受局部观测obs)

Actor 由表示层 `BasicMLP` 与输出层 `ActorNet` 级联组成：

- **表示层**：输入 `obs_dim + n_agents`，通过 `len(actor_hidden)` 个 `Linear+ReLU` 块映射
- **输出层**：单层 `Linear`，将表示映射为 `n_actions` 维 logits

$$
\text{input}_i^{(a)} = [o_i, \text{onehot}(i)]
$$

设隐藏层维度序列为 $H_a = [h_a^{(1)}, \dots, h_a^{(L)}]$，则：

$$
h_a^{(0)} = \text{input}_i^{(a)},\qquad
h_a^{(k)} = \text{ReLU}(W_a^{(k)} h_a^{(k-1)} + b_a^{(k)}),\ k=1,\dots,L
$$

$$
\text{logits}_i = W_a^{(L+1)} h_a^{(L)} + b_a^{(L+1)},\qquad
\pi_\theta(a_i|o_i, \text{id}_i) = \text{softmax}(\text{logits}_i)
$$

默认配置 $H_a = [64]$（单隐藏层 64 维）。

### Critic 网络 (接受全局状态state)

Critic 由表示层 `BasicMLP` 与输出层 `CriticNet` 级联组成：

- **表示层**：输入 `state_dim + n_agents`，通过 `len(critic_hidden)` 个 `Linear+ReLU` 块映射
- **输出层**：单层 `Linear`，输出标量状态价值

$$
\text{input}_i^{(c)} = [s, \text{onehot}(i)], \quad s = [o_1, o_2, \dots, o_N]
$$

设隐藏层维度序列为 $H_c = [h_c^{(1)}, \dots, h_c^{(M)}]$，则：

$$
h_c^{(0)} = \text{input}_i^{(c)},\qquad
h_c^{(k)} = \text{ReLU}(W_c^{(k)} h_c^{(k-1)} + b_c^{(k)}),\ k=1,\dots,M
$$

$$
V_\phi(s, i) = W_c^{(M+1)} h_c^{(M)} + b_c^{(M+1)}
$$

默认配置 $H_c = [256]$（单隐藏层 256 维）。

## 关键技术

### GAE (Generalized Advantage Estimation)

使用时序差分误差 $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$ 逐时间步反向传播计算优势函数，平衡偏差与方差：

$$
A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}
$$

其中 $\gamma$ 为折扣因子，$\lambda$ 为 GAE 平滑参数。

### PPO Clipped Objective

截断重要性采样比率，防止策略更新步长过大。Actor 损失函数为：

$$
L^{actor}(\theta) = -\mathbb{E}_t\left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为重要性采样比率，$\epsilon$ 为截断范围。

### Value Clipping

对价值函数输出做裁剪，防止价值估计剧烈波动。Critic 损失函数为：

$$
L^{critic}(\phi) = \mathbb{E}_t\left[ \max\left( (V_\phi(s_t) - R_t)^2,\; (V_\phi^{clip}(s_t) - R_t)^2 \right) \right]
$$

其中 $V_\phi^{clip}(s_t) = V_{\phi_{old}}(s_t) + \text{clip}(V_\phi(s_t) - V_{\phi_{old}}(s_t), -\epsilon, \epsilon)$。


## 训练流程

### 经验收集

每个 episode 开始时重置环境，之后每步执行：

1. **批维度扩展**：`obs (n_agents, obs_dim)` → `obs_batch (1, n_agents, obs_dim)`，state 同理
2. **联合前向**：调用 `get_actions(obs_batch, state_batch)`，同时计算：
   - Actor：拼接 one-hot agent ID → `BasicMLP` → `ActorNet` → Categorical 分布 → 采样 $a_{i,t}$，记录 $\log\pi_\theta(a_{i,t}|\cdot)$
   - Critic：state 复制平铺为 `(n_agents, state_dim)`，拼接 agent ID → `BasicMLP` → `CriticNet` → $V_\phi(s_t, \text{id}_i)$
3. **环境交互**：执行联合动作 $\mathbf{a}_t$，得到奖励 $\mathbf{r}_t$、下一观测 $\mathbf{o}_{t+1}$、终止标志 $d_t$
4. **存入 buffer**：`store(o_t, a_t, logp_t, r_t, d_t, V_t, s_t)`

重复直到 $d_t = \text{True}$ 或达到 `max_episode_steps`。

### Episode 结束处理

终止时触发 GAE 计算和 PPO 更新：

1. **last_value 计算**：
   - 若 $d_t = \text{True}$（自然终止）：`last_value = 0`
   - 若因最大步数截断：对 $s'$ 调用 `get_V` 计算 $V(s')$
2. **GAE 计算**（两步反向递推）：
   - 第一步：$\delta_{i,t} = r_{i,t} + \gamma V_{i,t+1}(1-d_t) - V_{i,t}$，令 $A_{i,t} = \delta_{i,t}$
   - 第二步：$A_{i,t} \mathrel{+}= \gamma\lambda(1-d_t) A_{i,t+1}$
   - 回报：$R_{i,t} = A_{i,t} + V_{i,t}$
3. **多轮 PPO 更新**：对 buffer 执行 `n_epochs` 轮

### PPO 更新

每轮 epoch 将索引随机打乱，按 `batch_size = buffer_size // n_minibatch` 切分 mini-batch。对每个 batch 执行以下步骤：

**1. 展平为 per-agent 形式**

所有张量 reshape 为 `(batch_size * n_agents, -1)`，Critic 将 state 复制平铺后拼接 agent ID。

**2. 优势归一化**

$$
\hat{A} = \frac{A - \mu_A}{\sigma_A + 10^{-8}}
$$

**3. 重要性采样比率**

$$
r_t(\theta) = \exp\left( \log\pi_\theta(a_t|\cdot) - \log\pi_{\theta_{old}}(a_t|\cdot) \right)
$$

**4. Actor 损失（CLIP）**

$$
L^{actor}(\theta) = -\mathbb{E}\left[ \min\left( r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]
$$

**5. Critic 损失（带裁剪，可选 Huber）**

$$
V_\phi^{clip} = V_{\phi_{old}} + \text{clip}(V_\phi - V_{\phi_{old}}, -\epsilon_v, \epsilon_v)
$$

$$
\ell(x, y) = \begin{cases}
\frac{1}{2}(x - y)^2 & |x - y| \leq \delta \\
\delta(|x - y| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

$$
L^{critic}(\phi) = \begin{cases}
\mathbb{E}\left[\max\left( \ell(V_\phi, R),\; \ell(V_\phi^{clip}, R) \right)\right] & \text{use\_value\_clip=True} \\[4pt]
\mathbb{E}\left[ \ell(V_\phi, R) \right] & \text{use\_value\_clip=False}
\end{cases}
$$

当 `use_huber_loss=False` 时 $\ell(x,y) = (x-y)^2$。

**6. 熵正则**

$$
\mathcal{H}(\pi_\theta(\cdot|s_t)) = -\sum_a \pi_\theta(a|s_t) \log\pi_\theta(a|s_t)
$$

**7. 总损失 & 参数更新**

$$
L = L^{actor} + c_v L^{critic} - c_e \mathcal{H}
$$

- 反向传播，计算梯度
- 若 `use_grad_clip=True`：`clip_grad_norm_(params, grad_clip_norm)`
- 优化器步进
- 若 `use_linear_lr_decay=True`：学习率调度器步进

更新后清空 buffer，进入下一 episode。（mappo是on-policy算法,episode结束，buffer里面的数据就失效了）

### 推理

推理使用确定性策略，仅 Actor 参与前向，Critic 和 buffer 均不涉及。

1. **加载模型**（可选）：`load_model(path)` 加载已训练的策略网络

2. **重置环境**：`obs = env.reset()`，获取初始观测 $\mathbf{o}_0 = [o_{1,0}, \dots, o_{N,0}]$

3. **每一步 $t$**：

   a. **添加 batch 维度**：$\mathbf{o}_t = (N, d_o) \to \mathbf{o}_t^{\text{batch}} = (1, N, d_o)$

   b. **生成 one-hot agent ID**：
   $$
   \text{id}_i = \text{onehot}(i) \in \mathbb{R}^N,\qquad
   \mathbf{ids} = [\text{id}_1; \dots; \text{id}_N] \in \mathbb{R}^{1 \times N \times N}
   $$

   c. **Actor 前向传播**：（仅调用ActorNet，不调用CriticNet）
   $$
   \mathbf{x}_i = [o_{i,t}, \text{id}_i]
   $$
   $$
   \mathbf{h}_i = \text{BasicMLP}_{\text{actor}}(\mathbf{x}_i),\qquad
   \text{logits}_i = \text{ActorNet}(\mathbf{h}_i)
   $$

   d. **选取最大概率动作**：
   $$
   a_{i,t}^* = \arg\max_a \, \text{softmax}(\text{logits}_i)_a = \arg\max_a \, \text{logits}_{i,a}
   $$

   e. **执行动作**：$\mathbf{a}_t^* = (a_{1,t}^*, \dots, a_{N,t}^*)$，调用 `env.step()` 得到下一观测 $\mathbf{o}_{t+1}$

   f. **渲染**（若 `render_mode` 非 None）：`env.render()`

4. **终止判断**：重复步骤 3 直到 $d_t = \text{True}$ 或达到 `max_episode_steps`


## 总损失函数

$$
L(\theta, \phi) = \underbrace{\mathbb{E}_t\left[L^{clip}(\theta)\right]}_{\text{Actor}} + c_v \underbrace{\mathbb{E}_t\left[L^{clip}(\phi)\right]}_{\text{Critic}} - c_e \underbrace{\mathbb{E}_t\left[\mathcal{H}(\pi_\theta(\cdot|s_t))\right]}_{\text{熵正则}}
$$

其中 $c_v = 0.5$ 为价值系数，$c_e = 0.01$ 为熵系数，$\mathcal{H}(\pi) = -\sum_a \pi(a)\log\pi(a)$ 为策略熵。

| 分量 | 公式 | 作用 |
|------|------|------|
| Actor | $-\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}\hat{A},\; \text{clip}(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1\pm\epsilon)\hat{A}\right)\right]$ | 限制策略更新步长 |
| Critic | $\mathbb{E}\left[\max\left(\ell(V_\phi, R),\; \ell(V_\phi^{clip}, R)\right)\right]$ | 稳定价值估计（$\ell$ 可选 MSE 或 Huber） |
| Entropy | $-\mathbb{E}\left[\mathcal{H}(\pi_\theta)\right]$ | 鼓励探索 |

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor_hidden_size` | `[64]` | Actor 隐藏层维度 |
| `critic_hidden_size` | `[256]` | Critic 隐藏层维度（MAPPO 集中式 Critic 需要更大容量） |
| `buffer_size` | 80000 | On-policy 缓存大小 |
| `n_minibatch` | 1 | 每 epoch 的 mini-batch 数（`batch_size = buffer_size // n_minibatch`） |
| `learning_rate` | 7e-4 | 学习率 |
| `gamma` | 0.95 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_range` | 0.2 | PPO 重要性采样比率截断范围 |
| `value_clip_range` | 0.2 | Critic 价值截断范围 |
| `vf_coef` | 0.5 | 价值损失系数 |
| `ent_coef` | 0.01 | 熵正则系数 |
| `n_epochs` | 10 | 每批数据训练轮数 |
| `use_huber_loss` | True | 是否使用 Huber 损失代替 MSE |
| `huber_delta` | 10.0 | Huber 损失的 δ 阈值 |
| `use_linear_lr_decay` | False | 是否启用线性学习率衰减 |
| `use_grad_clip` | True | 是否启用梯度裁剪 |
| `grad_clip_norm` | 10.0 | 梯度裁剪阈值 |

## 环境信息

- **任务**：`rware-tiny-4ag-v2`（Robotic Warehouse，4 个智能体）
- **动作空间**：5 个离散动作（前进、左转、右转、停留、装载）
- **观测维度**：71
- **全局状态**：所有智能体观测拼接（`obs_dim × n_agents = 284`）


