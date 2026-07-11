# RWARE 多智能体仓库环境 —— 新手入门指南

## 目录

- [RWARE 多智能体仓库环境 —— 新手入门指南](#rware-多智能体仓库环境--新手入门指南)
  - [目录](#目录)
  - [1. 简介](#1-简介)
    - [核心枚举](#核心枚举)
  - [2. 注册环境 ID](#2-注册环境-id)
    - [基础环境（自动注册）](#基础环境自动注册)
    - [图像观测变体](#图像观测变体)
    - [自定义列高变体](#自定义列高变体)
    - [完全自定义网格](#完全自定义网格)
  - [3. 自定义环境参数](#3-自定义环境参数)
  - [4. 动作与观测空间](#4-动作与观测空间)
    - [动作空间](#动作空间)
    - [观测空间（默认 FLATTENED）](#观测空间默认-flattened)
  - [5. Reward 机制](#5-reward-机制)
    - [默认奖励机制](#默认奖励机制)
    - [自定义奖励机制](#自定义奖励机制)
  - [6. Human Play 交互](#6-human-play-交互)
    - [启动](#启动)
    - [按键操作](#按键操作)
    - [运行示例](#运行示例)
    - [重要说明](#重要说明)
  - [7. 注意点（Tips）](#7-注意点tips)
  - [8. pip install 2.0.0版本 rware 库 Bug](#8-pip-install-200版本-rware-库-bug)
    - [问题](#问题)
    - [原因](#原因)
    - [修复操作](#修复操作)
    - [验证修复](#验证修复)

## 1. 简介

**RWARE**（Robotic Warehouse）是一个基于 Gymnasium 的多智能体仓库机器人模拟环境。

- **目标**：控制多个机器人在网格仓库中将指定货架搬运到目标区域（标记 `G`），每成功送达一个货架获得奖励
- **环境尺寸公式**（`tiny/small/medium/large`）：`高 = (column_height + 1) * shelf_rows + 2`，`宽 = 3 * shelf_columns + 1`
- **仓库布局示意图**：

```
----------
----------
-XX----XX- 
-XX----XX- 
-XX----XX-   <\
-XX----XX-   <- Shelf Rows
-XX----XX-   </
----------
----GG----
```

- **G** = 目标（Goal）位置，送达货架到此处获得奖励
- **XX** = 货架位置,如果有货物则可以抬起； **--** = 走廊/高速路（可通行）

### 核心枚举

| 枚举 | 值 |
|------|-----|
| `Action.NOOP = 0` | 什么都不做 |
| `Action.FORWARD = 1` | 前进 |
| `Action.LEFT = 2` | 左转 |
| `Action.RIGHT = 3` | 右转 |
| `Action.TOGGLE_LOAD = 4` | 装载/放下货架 |
| `Direction.UP = 0` | 上 |
| `Direction.DOWN = 1` | 下 |
| `Direction.LEFT = 2` | 左 |
| `Direction.RIGHT = 3` | 右 |

---

## 2. 注册环境 ID 

所有环境默认 `max_steps=500`，`column_height=8`，`sensor_range=1`，`msg_bits=0`。

### 基础环境（自动注册）

```
rware-{size}-{agents}ag{-difficulty}-v2
```

| 参数 | 可选值 |
|------|--------|
| `size` | `tiny` (1×3), `small` (2×3), `medium` (2×5), `large` (3×5) |
| `agents` | `1` ~ `19` |
| `-difficulty` | `-easy`（请求队列 = agents × 2），`-hard`（× 0.5），`空`（× 1） |

**示例**：
- `rware-tiny-2ag-v2` — 2 个 agent，小型仓库
- `rware-large-8ag-hard-v2` — 8 个 agent，大型仓库，困难模式
- `rware-small-4ag-easy-v2` — 4 个 agent，中尺寸，简单模式

### 图像观测变体

```
rware{-img/-imgdict}{-Nd}-{size}-{agents}ag{-difficulty}-v2
```
- `-img`：图像观测（多通道）
- `-imgdict`：图像 + 特征向量
- `-Nd`：非方向性（默认方向性，即图像随 agent 朝向旋转）

### 自定义列高变体

```
rware{-img}{-Nd}{-2s~-5s}-{size}-{column_height}h-{agents}ag{-difficulty}-v2
```
- `{-2s~-5s}`：传感器范围（默认 1）
- `{column_height}h`：货架列高度，范围 1~15

### 完全自定义网格

```
rware{-img}{-Nd}{-2s~-5s}-{rows}x{cols}-{column_height}h-{agents}ag-{req}req-{reward_type}-v2
```
- `rows`：1~4，`cols`：3/5/7/9
- `req`：请求队列大小 1~19
- `reward_type`：`indiv` / `global` / `twostage`

---

## 3. 自定义环境参数

```python
from rware.warehouse import Warehouse, RewardType, ObservationType

env = Warehouse(
    shelf_columns=3,           # 货架列数（必须奇数）
    column_height=8,           # 货架高度
    shelf_rows=1,              # 货架行数
    n_agents=2,                # 机器人数量
    msg_bits=0,                # 通信比特数（0 = 无通信）
    sensor_range=1,            # 观测半径
    request_queue_size=4,      # 同时请求的货架数
    max_inactivity_steps=None, # 最大无操作步数（None=无限）
    max_steps=500,             # 每回合最大步数
    reward_type=RewardType.INDIVIDUAL,
    observation_type=ObservationType.FLATTENED,
    render_mode="human",
)
```

或通过 `gym.make` 传入 kwargs：

```python
import gymnasium as gym
env = gym.make(
    "rware-tiny-2ag-v2",
    max_steps=500,
    reward_type=RewardType.GLOBAL,
    sensor_range=2,
)
```

---

## 4. 动作与观测空间

### 动作空间
rware\warehouse.py
```python
        class Action(Enum):
            NOOP = 0
            FORWARD = 1
            LEFT = 2
            RIGHT = 3
            TOGGLE_LOAD = 4  # 装载/放下货架
```

### 观测空间（默认 FLATTENED）

每个 agent 的观测是一个展平的一维向量，包含：

| 字段 | 长度 | 说明 |
|------|------|------|
| `self` 信息 | 7 | (x, y, carrying_shelf, direction_onehot×4, on_highway) |
| 传感器区域 | `(1+2×sensor_range)² × 7` | 每个感知格点：(has_agent, dir_onehot×4, msg?, has_shelf, shelf_requested) |
| **总计** | `7 + (1+2r)² × 7` | `r=1` 时为 70 维 |

若启用 `observation_type=ObservationType.DICT`，则返回结构化字典：

```python
{
    "self": {"location": [x, y], "carrying_shelf": [0/1], "direction": 0~3, "on_highway": [0/1]},
    "sensors": Tuple of {"has_agent", "direction", "has_shelf", "shelf_requested", ...}
}
```

---

## 5. Reward 机制

### 默认奖励机制

`step()` 返回 `rewards: List[float]`，长度 = `n_agents`。

| 奖励类型 | 触发条件 | 奖金 |
|----------|---------|------|
| `RewardType.INDIVIDUAL` | 某 agent 将请求中的货架送达 goal | 该 agent +1 |
| `RewardType.GLOBAL` | 任意 agent 送达货架 | 所有 agent +1 |
| `RewardType.TWO_STAGE` | 第一阶段：放下货架于非高速路 | +0.5 |
| | 第二阶段：送达至 goal | +0.5 |

**回合终止条件**：
- `_cur_steps >= max_steps`（默认 500）
- `max_inactivity_steps` 不为 None 且连续无送达步数超过限制

### 自定义奖励机制

参见 `rware_human_play-modify-reward.py`

---

## 6. Human Play 交互

### 启动

```bash
conda activate rlrl
python human_play.py --env rware-tiny-2ag-v2 --max_steps 500
```

可选参数：
- `--env`：环境 ID（默认 `rware-tiny-2ag-v2`）
- `--max_steps`：最大步数（默认 500）
- `--display_info`：开启详细信息显示（观测、奖励、位置等）

### 按键操作

| 按键 | 功能 |
|------|------|
| `↑` | 当前 agent 前进 |
| `←` / `→` | 当前 agent 左转 / 右转 |
| `P` / `L` | 装载 / 放下货架 |
| `SPACE` | 什么都不做（NOOP） |
| `TAB` | 切换到下一个 agent |
| `R` | 重置环境 |
| `H` | 显示帮助 |
| `D` | 切换详细调试信息 |
| `ESC` | 退出 |

### 运行示例

```
Environment: rware-tiny-2ag-v2
Max steps per episode: 500
Number of agents: 2
Action space: Tuple(Discrete(5), Discrete(5))
Observation space: Tuple(Box([-inf ...], [inf ...], (70,), float32), ...)

Step 1: Agent 1 reward = 0  (cumulative: [0.0, 0.0])
Step 2: Agent 1 reward = 0  (cumulative: [0.0, 0.0])
Step 3: Agent 1 reward = 0  (cumulative: [0.0, 0.0])
```

### 重要说明

**当前 `human_play.py` 是轮流控制模式**：每次按键只控制当前选中的 agent，其他 agent 在本步执行 `NOOP`。

这意味着每一个 step 只有一个 agent 在执行动作，其余原地等待。这是交互式 demo 的简化设计，并非为每个 agent 分配独立按键。

如果要在脚本中实现所有 agent 同时行动，需调用：
```python
actions = [Action.FORWARD, Action.LEFT]  # 每个 agent 一个动作
obs, rews, done, trunc, info = env.step([a.value for a in actions])
```
---
## 7. 注意点（Tips）

- **所有 agent 同时 `step()`**：环境核心逻辑要求传入一个长度为 `n_agents` 的动作列表，所有 agent **同步**执行动作，不存在轮流概念。`human_play.py` 只是简化 demo
- **碰撞处理**：环境使用有向图 + 拓扑排序自动解决 agent 间的移动冲突，具有冲突避免机制
- **通信**：`msg_bits > 0` 时，agent 在动作中可附带消息（动作变为 `MultiDiscrete([5, 2, 2, ...])`），观测中也包含邻居消息
- **Image 观测**：`-img` 模式返回 `(C, H, W)` 格式的多通道图像，各个通道由 `ImageLayer` 枚举控制
- **自定义布局**：在 `Warehouse.__init__` 中传入 `layout` 字符串（`X`=货架，`.`=走廊，`G`=目标），可完全自定义仓库地图

---

## 8. pip install 2.0.0版本 rware 库 Bug

使用pip install 2.0.0版本会有以下错误：直接从github仓库中安装2.0.0则不会有

### 问题

运行 `human_play.py` 时出现：

```
AttributeError: 'Warehouse' object has no attribute 'shelfs'
```

### 原因

在 `rware/warehouse.py` 的 `reset()` 方法中，`self.render()` 在第 763 行被调用，但此时 `self.shelfs`（第 774 行）和 `self.agents`（第 789 行）尚未初始化。渲染函数 `_draw_shelfs` 访问 `env.shelfs` 导致崩溃。

### 修复操作

编辑文件 `D:\.conda\envs\rlrl\Lib\site-packages\rware\warehouse.py`，

将 `reset()` 方法中第 **762-763 行**的渲染调用移动到 shelfs 和 agents 初始化**之后**：

```python
# 修改前（~line 760-765）：
super().reset(seed=seed, options=options)
if self.render_mode == "human":
    self.render()           # <-- BUG: 此时 shelfs 和 agents 尚未创建
Shelf.counter = 0
Agent.counter = 0
...

# 修改后：
super().reset(seed=seed, options=options)
Shelf.counter = 0
Agent.counter = 0
...
self.shelfs = [...]          # shelfs 初始化
self.agents = [...]          # agents 初始化
if self.render_mode == "human":
    self.render()            # <-- FIX: 移到 shelfs/agents 之后
```

### 验证修复

```bash
conda activate rlrl
python human_play.py --help
# 如果能正常显示帮助信息，说明修复生效
```



