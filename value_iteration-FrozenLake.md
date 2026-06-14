# value_iteration(原始值迭代)

$Q_k(s,a)=\sum_{s'} p(s'\mid s,a) [r + \gamma v_{k-1}(s')]$

$V_{k+1}(s)=\max_{a} Q_k(s,a)$

$\pi_{k+1}(s)=\argmax_{a} Q_k(s,a)$