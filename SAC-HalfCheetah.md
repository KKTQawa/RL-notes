
# SAC

## act
高斯重采样

$u=\mu+\epsilon\sigma ,\epsilon \sim N(0, I)$

压缩$a\in [-1, 1]$范围

$a=\tanh(u)$

数值稳定

$log\pi(a)=\log p(u)-\log(1-\tanh^2(u))= \log p(u)-2*\log2-u-softplus(-2u)$

## train

### 1. 从经验池里面抽一批数据

### 2. 目标网络的“损失”-TD-error

$y_{target}=r+\gamma( \min_{i=1,2} Q_{target_i}(s',a')-\alpha * log\pi(a'|s'))$

>原始TD-error公式

$Q_{\pi}(s,a)=\mathbb E_{\pi} [r+\gamma Q_{\pi}(s',a')]$

### 3. 真实网络的损失

$L_{current}=(Q_{current_1}-y_{target})^2+(Q_{current_2}-y_{target})^2$

同时反向传播更新current_1和current_2的参数

$\theta_{current} = \theta_{current} - \alpha_{learning} * \nabla_{\theta_{current}} L_{current}$,$\alpha_{learning}$是超参数，学习率

然后更新target网络的参数

- update_target

$\theta_{target} = \tau * \theta_{current} + (1 - \tau) * \theta_{target}$,$\tau$是超参数，网络的学习率
### 4. actor_loss

$L_{actor}=\mathbb E(\alpha log\pi(a|s)-\min(Q_{current_1}(s,a),Q_{current_2}(s,a)))$

- 这里的$\mathbb E$是期望，是通过经验回放池里面随机抽取的批次样本来计算得到的

### 5. alpha_loss

$L_{alpha}=\mathbb E(-\log\alpha(\log\pi(a|s)+H_0)),取H_0\approx action_dim$

## 深度推导

>总目标最大化

$J(\pi)=\mathbb E_t[L_t]=\mathbb E_t[\gamma^t(r_t+\alpha \mathcal H)]$

同时保证

$\mathcal H>= H_0,\mathcal H=-\mathbb E_{\pi} log\pi(a|s)$


$\therefore J(\pi)=\mathbb E_t[L_t]=\mathbb E_t[\gamma^t(r_t-\alpha log\pi(a|s))]$

### actor_loss来源

$\because 最优策略\pi^*(a|s)\propto exp(\frac{1}{\alpha}Q(s,a))$

$\therefore Q(s,a)-\alpha log\pi(a|s)=C$

所以取动作策略优化目标为

$L_{actor}=\mathbb E[\alpha log\pi(a|s)-Q(s,a)]$

> 在最大熵强化学习框架中，最优策略具有玻尔兹曼分布形式：

$$
\pi^*(a|s) = \frac{\exp\left(\frac{1}{\alpha} Q(s,a)\right)}{\int \exp\left(\frac{1}{\alpha} Q(s,a')\right) da'}
$$

其中：
- $Q(s,a)$ 是软动作价值函数
- $\alpha$ 是温度参数，控制熵正则项的强度
- 分母是配分函数 $Z(s)$，用于归一化

>参数化策略 $\pi_\phi$ 应该尽量接近这个最优分布

$$
\min_\phi D_{\text{KL}}\left(\pi_\phi(\cdot|s) \;\middle\|\; \pi^*(\cdot|s)\right)
$$

$$
\begin{aligned}
D_{\text{KL}}\left(\pi_\phi \| \pi^*\right) 
&= \mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi(a|s) - \log \pi^*(a|s)\right] \\
&= \mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi(a|s) - \frac{1}{\alpha}Q(s,a) + \log Z(s)\right]\\
&\stackrel{忽略无关项}{==} \mathbb{E}_{a \sim \pi_\phi}\left[\log \pi_\phi(a|s) - \frac{1}{\alpha}Q(s,a)\right]
\end{aligned}
$$

其中 $\log Z(s)$ 是配分函数的对数，与策略参数 $\phi$ 无关。

>进一步地

$$
\min_\phi \mathbb{E}_{a \sim \pi_\phi}\left[\alpha \log \pi_\phi(a|s) - Q(s,a)\right]
$$


### alpha_loss来源
给定约束条件下，$J(\pi)$最大化同时就必须最小化

$-\mathbb E[\log\pi(a|s)+H_0]$

借鉴拉格朗日乘子法思想，添加参数$\alpha$,并且迭代过程为了保证$\alpha>0$，令

$\alpha=e^{log\alpha}>0$
