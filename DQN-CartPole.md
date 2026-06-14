
# 调整奖励函数
r1=$1-\frac{|x|}{x_{max}}-0.8$

r2=$1-\frac{|\theta|}{\theta_{max}}-0.5$

r=r1+r2

#  Loss Function

使用均方误差：

$$
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
Q(s_i, a_i; \theta) - y_i
\right)^2
$$

# eval Network Update

梯度下降更新：

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

#  Target Network Update

每隔 K 步进行硬更新：

$$
\theta^- \leftarrow \theta
$$

#  Full DQN Objective

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\left(
Q(s,a;\theta)
-
\left(
r + \gamma \max_{a'} Q(s',a';\theta^-)
\right)
\right)^2
\right]
$$