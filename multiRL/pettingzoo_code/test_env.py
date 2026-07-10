from mpe2 import simple_push_v3
from shimmy import OpenSpielCompatibilityV0
from pettingzoo.butterfly import pistonball_v6
import os
ENV_LISTS_CATEGORY = [
    "Simple",
    "Simple Adversary",
    "Simple Crypto",
    "Simple Formation",
    "Simple Line",
    "Simple Push",
    "Simple Reference",
    "Simple Speaker Listener",
    "Simple Spread",
    "Simple Tag",
    "Simple World Comm",
    "Collect Treasure"
]
ENV_LISTS=[
    "collect_treasure_v1",         
    "simple_adversary_v3",           
    "simple_crypto_v3",            
    "simple_formation_v1",         
    "simple_line_v1",               
    "simple_push_v3",                
    "simple_reference_v3",           
    "simple_speaker_listener_v4",  
    "simple_spread_v3",               
    "simple_tag_v3",                
    "simple_v3",                
    "simple_world_comm_v3",          
]                                  
# from pettingzoo.butterfly import (
#     cooperative_pong_v6,
#     knights_archers_zombies_v10,
#     pistonball_v6,
# )
# from pettingzoo.classic import (
#     chess_v6,
#     connect_four_v3,
#     gin_rummy_v4,
#     go_v5,
#     hanabi_v5,
#     leduc_holdem_v4,
#     rps_v2,
#     texas_holdem_no_limit_v6,
#     texas_holdem_v4,
#     tictactoe_v3,
# )
# from pettingzoo.sisl import multiwalker_v9, pursuit_v4

env_id = "pistonball_v6"
# observation_space: Box(0, 255, (457, 120, 3), uint8)
# action_space: Box(-1.0, 1.0, (1,), float32)
# agent: piston_0
def run_env(env_id):

    try:
        env = pistonball_v6.env(render_mode="human")
        env.reset(seed=42)

        ep=0
        for agent in env.agent_iter():
            ep+=1
            if ep==1:
                print(f"Running environment: {env_id}")
                print('observation_space:',env.observation_space(agent))
                print('action_space:',env.action_space(agent))  
                print("agent:",agent)
            observation, reward, termination, truncation, info = env.last()
            print("reward:",reward)

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = env.action_space(agent).sample()

            env.step(action)

    except KeyboardInterrupt:
        print("\nExiting:", env_id)

    env.close()

if __name__ == "__main__":
    run_env(env_id)
