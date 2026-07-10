# VDN
所有智能体共享一个Q网络（参数共享）
## 1. Q_total

将所有智能体的Q值求和：

$$Q_{total} = \sum_{i=1}^{N} Q(s, a)$$

## 2. target Q

目标总Q值：

$$Q_{target} = r + \gamma \cdot (1 - d) \cdot \sum_{i=1}^{N} \max_{a'} Q(s_{next},a')$$

其中：
- $r$：奖励向量
- $d$：终止标志 (0或1),假如终止则 $Q_{target}=r$,没有后续
- $\gamma$：折扣因子 (0.99)
- Q和Q_target网络输出的是一个动作集合的概率分布,输入是state

## 3. 损失函数

MSE损失：

$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \left( Q_{total} - Q_{target} \right)^2$$

或：

$$\mathcal{L}(\theta) = \text{MSE}(\mathbf{Q}_{\text{total}}, \mathbf{Q}_{\text{target}})$$

## 4. 参数更新
原始Q网络：

$$\theta \leftarrow \theta - \alpha \cdot \nabla_{\theta} \mathcal{L}(\theta)$$

目标Q网络：每 $T$ 步拷贝原始网络参数：

$$\theta_{target} \leftarrow \theta$$





