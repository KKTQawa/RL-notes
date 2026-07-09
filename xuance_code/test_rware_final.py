"""Quick test to verify rware training works end-to-end."""
import sys
sys.path.insert(0, '.')
from algorithm import get_runner, train_rware, evaluate_rware

result = train_rware('qmix', 'rware-tiny-2ag-v2', training_steps=500, eval_interval=0, batch_size=16, buffer_size=2000)
print(f'Episodes completed: {len(result["train_rewards"])}')
if result['train_rewards']:
    print(f'Last 5: {[round(r, 2) for r in result["train_rewards"][-5:]]}')
    print(f'Avg: {sum(result["train_rewards"])/len(result["train_rewards"]):.3f}')

# Evaluate
runner = get_runner('qmix', 'robotic_warehouse', 'rware-tiny-2ag-v2', {'training_steps': 500, 'batch_size': 16, 'buffer_size': 2000})
runner.run('train')
eval_out = evaluate_rware(runner, n_episodes=5)
print(f'Eval reward: {eval_out["avg_reward"]:.3f}')
print('All OK!')
