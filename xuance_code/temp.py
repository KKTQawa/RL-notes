import os
import glob

# 搜索整个 conda 环境
env_path = r'D:\.conda\envs\xuance_env'
sdl_files = glob.glob(os.path.join(env_path, '**', 'SDL2.dll'), recursive=True)
print("找到的 SDL2.dll：")
for f in sdl_files:
    print(f"  {f}")

# 搜索系统 PATH 中的所有位置
import subprocess
result = subprocess.run(['where', 'SDL2.dll'], capture_output=True, text=True)
print("\n系统 PATH 中的 SDL2.dll：")
print(result.stdout)