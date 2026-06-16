# RL-notes

强化学习算法实现笔记，包含多种经典RL算法在不同环境中的实现。

每个算法-环境组合都有独立的脚本，便于理解算法流程和实验对比。

## 算法/环境列表

> 见文件列表

## 项目结构

```
.
├── README.md
├── all.txt                    # 包含所有gymnasium环境的id
├── test_env.py                # 用于快速查看环境信息
├── model/                     # 保存训练好的模型
├── saves/                     # 保存训练过程数据
├── *.py                       # 各算法实现脚本
├── *.md                       # 算法说明文档
├── *_train.png                # 训练曲线图
└── *_actor_net.pt             # 部分文件训练好的动作网络权重
```

## 安装

**环境要求**: Python 3.10左右

```bash
pip install gymnasium ale-py numpy torch matplotlib
```

部分环境模块
```bash
pip install "gymnasium[atari]"
pip install "gymnasium[box2d]"
pip install "gymnasium[mujoco]"

```
所有
```bash
pip install "gymnasium[all]"
```

## 使用方法

每个算法脚本独立运行

```bash

python DQN-CartPole.py

# 查看环境信息
python test_env.py
```


