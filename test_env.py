import gymnasium as gym
import os
import ale_py
import numpy as np
#封装函数
class WrappedEnv(gym.Wrapper):
    def __init__(self, env):
        super(WrappedEnv, self).__init__(env)
        self.true_env = env
        #解包
        while hasattr(self.true_env, "env"):
            self.true_env = self.true_env.env
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # 调整奖励

        return observation, reward, terminated, truncated, info
    
# for env_id in gym.envs.registry.keys():
#     print(env_id)

# 创建原始环境
gym.register_envs(ale_py)
env = gym.make('ALE/Backgammon-v5', render_mode="human")#human表示渲染模式为human，即在屏幕上显示环境状态
print('env.spec:',env.spec)
print('observation_space:',env.observation_space)
print('action_space:',env.action_space)
# 封装环境
wrapped_env = WrappedEnv(env)

# 使用封装后的环境进行交互
observation, info = wrapped_env.reset()
terminated=False
truncated=False
i=0
while not truncated and not terminated:
    #动作空间随机选择
    action = wrapped_env.action_space.sample()
    print('action:', action)
    #ac=float(input("please input action:during -3 to 3\n"))
    #action = np.array([ac], dtype=np.float32)
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f'>>>{i}>>>')
    i+=1
    print('observation:', observation)
    print('reward:', reward)
    print('terminated:', terminated)

os.system("pause")
wrapped_env.close()