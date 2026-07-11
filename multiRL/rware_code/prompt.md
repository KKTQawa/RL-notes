# 概述

xuance是一个多智能体RL库，内置了很多的算法实现。但是模块化的文件不方便学习算法，现在我想学习iql算法，并且在rware这个环境的'iql-tiny-4ag-v2'任务上进行模型训练、推理。你需要在当前目录下创建iql-rwarehouse.py文件，实现上述功能。最后，再创建iql.md文件介绍这个算法的架构、训练、推理过程

# 注意事项

- 你需要确保已经激活了conda的xuance_env环境
- 你只能修改你创建的文件
- 务必参考xuance库的源码：D:\.conda\envs\xuance_env\Lib\site-packages\xuance里面的文件。特别地，D:\.conda\envs\xuance_env\Lib\site-packages\xuance\configs\这个目录下你只需要参考D:\.conda\envs\xuance_env\Lib\site-packages\xuance\configs\iql\robotic_warehouse\rware-tiny-2ag-v2.yaml即可，其他的文件都无关
- 你不能import xuance,也就是说你的核心算法逻辑必须从xuance这个库里面提取
- 请不要编写脚本完成任务
- 完成后，务必运行代码，保证代码能够正常运行
