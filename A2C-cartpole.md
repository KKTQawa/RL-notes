# A2C

## 1.等待一个episode结束

## 2.计算时序差分目标（TD Target）和优势估计

$Q_{target\; t+1}'(s_{t+1})$

$y_t = r_t + \gamma*(1-done_t) * Q_{target\; t+1}'(s_{t+1})$

$A_t = y_t - Q_{t}(s_t)$

## 3.Actor网络更新

$a_{\text{Loss}} = -\sum_{N} \log \pi(a|s) * A_t$

## 4.Critic网络更新

$\text{Critic}_{\text{Loss}} = \text{MSE} (y_t\; ,Q_{t}(s_t))$

## 5.延迟更新目标网络(每隔几步更新一次)

$\theta^-_{target} = \theta_{current}$




