
# PPO

## 1. 采样轨迹
在旧策略 $\pi_{\theta_{\text{old}}}$ 下与环境交互完整一轮，获取经验数据
>动作修正

$a=tanh(u)$

$\log \pi(a|s) = \log \pi(u|s)-\log(1-\tanh^2(u))$


## 2. 计算时序差分TD误差

$$\delta_t = r_t + \gamma (1 - d_t) V_{t+1} - V_t$$

## 3. 递归计算GAE优势函数 (Generalized Advantage Estimation)
>n步优势
$$A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}$$
>GAE通用形式
$$A_t=\sum_{l=0}^{\infty} (\gamma^l \lambda)^l \delta_{t+l}$$

>标准化

$$A_t = \frac{A_t - \mathbb{E}_t A_t}{\sqrt{\operatorname{Var}_t A_t + \epsilon}}$$

## 4.重复 $K$ 次PPO更新


### 4.1. 策略概率比 (Importance Sampling)

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \exp\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{\text{old}}}(a_t|s_t)\big)$$

### 4.2. Actor Loss

$$\mathcal{L}_{\text{actor}}=-\mathcal{L}^{\text{clip}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

>参数更新

$$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$$

### 4.3. Critic Loss
回报：
$$R_t = A_t + V_t$$

价值网络的均方误差损失：

$$\mathcal{L}_{\text{critic}} = \mathbb{E}_t \left[ \left( V_\phi(s_t) - R_t \right)^2 \right]$$

>参数更新
$$\phi \leftarrow \phi - \alpha \nabla_\phi \mathcal{L}$$

### 4.4. 熵奖励 (已省略)
策略熵正则项，鼓励探索：
$$\mathcal{L}_{\text{entropy}} = -\mathbb{E}_t \left[ H(\pi_\theta(\cdot|s_t)) \right]$$

### 4.5. 总损失
综合三项损失，其中 $c_1, c_2$ 为平衡系数：
$$\mathcal{L} = \mathcal{L}_{\text{actor}} + c_1 \mathcal{L}_{\text{critic}} - c_2 \mathcal{L}_{\text{entropy}}$$



