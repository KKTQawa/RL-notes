$a_t = 
\begin{cases}
\arg\max_{a} Q(s_t, a), & \text{以概率 } 1 - \epsilon \\
\text{随机动作}, & \text{以概率 } \epsilon
\end{cases}$

$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \bigr]$