# TD3

每步更新
## 1.步进

确定性策略+探索性噪声

$a_t = \pi(s_t)+N(0, noise_std²)$

## 2.从历史经验中采样单个批次

## 3.计算目标Q值

$Q_{target1}(s_{t+1}, a_{target\, t+1}')=\pi(a_{target1\; t+1}'|s_{t+1})$

$Q_{target2}(s_{t+1}, a_{target\, t+1}')=\pi(a_{target2\; t+1}'|s_{t+1})$

$Q_{target}(s_t, a_t) = r_t + γ * min(Q_{target1}(s_{t+1}, a_{target\, t+1}), Q_{target2}(s_{t+1}, a_{target\, t+1}))$ * (1 - done_t)$

## 4.更新Critic（Q网络）

$Q_{Loss_1} =\frac{1}{N} \sum_{t=1}^{N} (Q_{target}(s_t, a_t) - Q_1(s_t, a_t))^2$

$Q_{Loss_2} =\frac{1}{N} \sum_{t=1}^{N} (Q_{target}(s_t, a_t) - Q_2(s_t, a_t))^2$

## 5.延迟更新Actor（策略网络）（每隔几步更新一次）
>$Q_1$和$Q_2$网络随便取一个

$a_{Loss} = -\mathbb{E}_{N}(Q_1(s_t, \pi(s_t)))$

## 6.更新目标网络

$\theta^-_{actor} =\tau \theta_{actor} +(1-\tau)\theta^-_{actor}$

$\theta^-_{critic_1} =\tau \theta_{critic_1} +(1-\tau)\theta^-_{critic_1}$

$\theta^-_{critic_2} =\tau \theta_{critic_2} +(1-\tau)\theta^-_{critic_2}$



