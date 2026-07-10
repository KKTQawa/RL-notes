import xuance
from argparse import Namespace

def run(algo,env,env_id,mode,parser_args):
    runner = xuance.get_runner(algo=algo,
                           env=env,  # 选择：sc2, mpe, robotic_warehouse, football, magent2.
                           env_id=env_id,  # 选择：3m, 2m_vs_1z, 8m, 1c3s5z, 2s3z, 25m, 5m_vs_6m, 8m_vs_9m, MMM2 等。
                           parser_args=parser_args
                           )  # False 用于训练，True 用于测试
    runner.run(mode=mode)  # 开始运行（或 runner.benchmark() 用于基准测试）

if __name__ == '__main__':
    parser_args = Namespace(
    render=True,
    render_mode="human",
    running_steps=1000000,
    logger="tensorboard",#或者"wandb"
    video_dir="videos/maddpg/",
    max_episode_steps=500,
    parallels=1
    #device="cpu" ,
    #dl_toolbox="torch" 
)
    #run('iql','robotic_warehouse','rware-tiny-2ag-v2','test',parser_args)
    run ('qmix','robotic_warehouse','rware-tiny-2ag-v2','train',parser_args)
    #run ('wqmix','mpe','simple_spread_v3','train',parser_args)
    #run ('maddpg','mpe','simple_spread_v3','train',parser_args)
    #run ('vdn','robotic_warehouse','rware-tiny-2ag-v2','train',parser_args)
    #run ('ippo','mpe','simple_spread_v3','train',parser_args)
    #run ('mappo','robotic_warehouse','rware-tiny-2ag-v2','train',parser_args)
    #run ('coma','mpe','simple_spread_v3','train',parser_args)

