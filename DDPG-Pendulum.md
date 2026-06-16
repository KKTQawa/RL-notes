
# DDPG（连续动作空间）

每步更新
## 1.步进

确定性策略+探索性噪声

$a_t = \pi(s_t)+N(0, noise_std²)$

## 2.从历史经验中采样单个批次

## 3.计算目标Q值

$Q_{target}(s_t, a_t) = r_t + γ * Q(s_{t+1}, a_{target\, t+1}') * (1 - done_t)$

## 4.更新Critic（Q网络）

$Q_{Loss} =\frac{1}{N} \sum_{t=1}^{N} (Q_{target}(s_t, a_t) - Q(s_t, a_t))^2$

## 5.更新Actor（策略网络）

$a_{Loss} = -\mathbb{E}_{N}(Q(s_t, \pi(s_t)))$

## 6.更新目标网络

$\theta^-_{actor} =\tau \theta_{actor} +(1-\tau)\theta^-_{actor}$

$\theta^-_{critic} =\tau \theta_{critic} +(1-\tau)\theta^-_{critic}$
