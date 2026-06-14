import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import os

from tqdm import trange, tqdm#进度条
#Acrobot-v1
class EGreedyExpStrategy(): # epsilons-greedy strategy
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=1000000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, q_values, state):

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
            self.exploratory_action_taken = False
        else:
            action = np.random.randint(len(q_values))
            self.exploratory_action_taken = True

        self._epsilon_update()
        return action
    

# Import and initialize Mountain Car Environment
env = gym.make('Acrobot-v1')
env.reset()

DISCRETE_OS_SIZE = np.array([5,5,5,5,8,8])

discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE

def get_discrete_state(state):

    discrete_state = (
        state - env.observation_space.low
    ) / discrete_os_win_size

    discrete_state = discrete_state.astype(np.int32)

    # 防止越界
    discrete_state = np.clip(
        discrete_state,
        0,
        DISCRETE_OS_SIZE - 1
    )

    return tuple(discrete_state)
# Define Q-learning function
training_strategy = EGreedyExpStrategy(init_epsilon=0.8, min_epsilon=0.01, decay_steps=500000)

num_states = [5,5,5,5,8,8]

Q = np.random.uniform(
    low=-1,
    high=1,
    size=(num_states + [env.action_space.n])
)
def QLearning(env, learning, discount,episodes):

    # randomly initialize a Q table
    #Q = np.random.uniform(low=-2, high=0, size=(num_states + [env.action_space.n]))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    reach_rate = np.zeros(int(episodes/100))
    r = 0
    # Calculate episodic reduction in epsilon
    #training_strategy = EGreedyExpStrategy(init_epsilon=0.8, min_epsilon=0.01, decay_steps=500000)
    
    min_episode=500
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state, info = env.reset()
        
        # Discretize state
        state_adj = get_discrete_state(state)
        j=0
        while done != True:        
            j+=1       
            # Determine next action - epsilon greedy strategy
            action = training_strategy.select_action(Q[state_adj], state)
                
            # Get next state and reward
            new_state, reward, terminated, truncated, info = env.step(action)
            #print(f'episode {i+1} state {new_state} reward {reward}') #[position,velocity]

            done = terminated or truncated
                
            # Discretize new_state
            new_state_adj = get_discrete_state(new_state)
            #Allow for terminal states
            if done:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning * (
                    reward +
                    discount * np.max(Q[new_state_adj]) -
                    Q[state_adj + (action,)]
                )

                Q[state_adj + (action,)] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = new_state_adj
        if tot_reward != -500.0:#默认最大500步结束
            reach_rate[r] = reach_rate[r] + 1
            if j<min_episode:
                min_episode=j
                np.savez('Q_training_data.npz', Q=Q, epsilon=training_strategy.epsilon)
                print(f'第{i}轮：min_episode:{min_episode}')
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = [] 
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            print('成功次数/100轮:',reach_rate[r])
            r = r + 1
    print('final min_episode:',min_episode)
    env.close()
    
    return ave_reward_list, reach_rate

# Run Q-learning algorithm
rewards, reach = QLearning(env, 0.3, 0.99,500)

def run():
    env1 = gym.make('Acrobot-v1',render_mode="human")
    #读档
    data = np.load('Q_training_data.npz')
    Q = data['Q']
    epsilon = float(data['epsilon'])

    done = False
    state, info = env1.reset()
    state_adj = get_discrete_state(state)
    i=0
    while done != True:               
        i=i+1;
        action = training_strategy.select_action(Q[state_adj], state)  
        new_state, reward, terminated, truncated, info = env1.step(action)
        done = terminated or truncated
        state_adj = get_discrete_state(new_state)
    print('final episode:',i)
    os.system("pause")
    env1.close()

# print('final epsilon:',training_strategy.epsilon)
# print('final Q table',Q)
#保存参数
#np.savez('Q_training_data.npz', Q=Q, epsilon=training_strategy.epsilon)

plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.plot(100*(np.arange(len(reach)) + 1), reach)
plt.xlabel('Episodes')
plt.ylabel('Average Reward & Reach times')
plt.title('Average Reward & Reach times vs Episodes')
plt.savefig('rewards.jpg')     
plt.close()  
os.system('pause')

run()

