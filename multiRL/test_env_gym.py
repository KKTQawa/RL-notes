import gymnasium as gym

import os

ENV_LIST = [
"dm_control/acrobot-swingup_sparse-v0"
]

def run_env(env_id):
    env = gym.make(env_id, render_mode="human")

    for env_id in gym.envs.registry.keys():
        print(env_id)

    print(f"Running environment: {env_id}")
    print('env.spec:',env.spec)
    print('observation_space:',env.observation_space)
    print('action_space:',env.action_space)
    env.close()
    return
    obs, info = env.reset()

    try:
        while True:
            #action = env.action_space.sample()
            action=int(input("请输入动作: "))#0 -none,1 -shoot,2-right 3-left

            obs, reward, terminated, truncated, info = env.step(action)
            print("状态:",obs)
            print("奖励:",reward)

            if terminated or truncated:
                os.system("pause")
                obs, info = env.reset()
                break


    except KeyboardInterrupt:
        print("\nExiting:", env_id)

    env.close()


if __name__ == "__main__":
    for env_id in ENV_LIST:
        run_env(env_id)
    #run_env("ALE/Adventure-v5")
