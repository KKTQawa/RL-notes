import gymnasium as gym
from shimmy import GymV26CompatibilityV0

# 创建原始的 gfootball 环境（它仍然是旧版 gym 接口）
# 注意：这里用 gym.make，但实际返回的是旧版环境
raw_env = gym.make('GFootball/academy_3_vs_1_with_keeper-simplev1-v0', render_mode="human")  

# 用 Shimmy 包装器将其转换为 Gymnasium 接口
env = GymV26CompatibilityV0(env=raw_env, env_id="gfootball")

# 现在 env 就是标准的 Gymnasium 环境了
obs, info = env.reset(seed=42)
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
env.close()