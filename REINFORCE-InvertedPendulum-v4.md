
# Policy_Network
>这个更偏向于连续动作空间 Auxiliary Network

>使用同一个网络来估计$\mu$和$\sigma$，然后根据$\mu$和$\sigma$固定的高斯分布来采样动作

$h_1 = \tanh(W_1 s + b_1)$  

$h_2 = \tanh(W_2 h_1 + b_2)$

$\mu = W_\mu h_2 + b_\mu$

$\hat{\sigma} = W_\sigma h_2 + b_\sigma$

$\sigma = \log(1 + e^{\hat{\sigma}})$

$a \sim \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$

# 蒙特卡洛方法REINFORCE

折扣累计回报
$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{T-t-1} R_{T-1}
$$

损失函数
$$
L(\theta)
=
-
\sum_{t=0}^{T-1}
\log \pi_\theta(a_t \mid s_t)\, G_t
$$

*手写代码只需要写到这里就可以了，以下为pytorch自动实现*

策略梯度目标
$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_{t=0}^{T-1}
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)\, G_t
\right]
$$
参数更新公式
$$
\theta
\leftarrow
\theta
+
\alpha
\sum_{t=0}^{T-1}
\nabla_\theta
\log \pi_\theta(a_t \mid s_t)\, G_t
$$





