#DQN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os

BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
MEMORY_CAPACITY = 2000                  # The capacity of experience replay buffer

env = gym.make("CartPole-v0") # Use cartpole game as environment 坚持越久越好
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0] # 4 states
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):
    def __init__(self):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1) # 初始化  weights of fc1
        self.out = nn.Linear(10, N_ACTIONS) # layer 2
        self.out.weight.data.normal_(0, 0.1) # 初始化  weights of fc2
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
        
# 3. Define the DQN network and its corresponding methods
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0 # count the steps of learning process
        self.memory_counter = 0 # counter used for experience replay buffer
        
        # ----Define the memory (or the buffer), allocate some space to it. The number 
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) 
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
    
        self.loss_func = nn.MSELoss()#使用均方误差
        
    def  choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state x

        if np.random.uniform() < EPSILON:   
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)
            #print(torch.max(actions_value, 1)) 
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   
            # random
            action = np.random.randint(0, N_ACTIONS)
            #ENV_A_SHAPE=0为离散空间，ENV_A_SHAPE=(1,)为连续动作空间
            #action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    
    def predict(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        self.eval_net.eval()

        with torch.no_grad():
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].item()

        return action
        
    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer    
        # s_是执行a之后的下一个状态    
        transition = np.hstack((s, [a, r], s_)) # horizontally stack these vectors
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self):

        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            # target_net直接照抄eval_net的参数
            # 每隔一定次数，否则容易震荡
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # randomly select some data from buffer
        b_memory = self.memory[sample_index, :]

        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        
        #神经网络支持Batch输入
        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a) # gather表示沿着第一维度取值 gather(dim, index),b_a=0/1
        #print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach() # detach from computational graph, don't back propagate
        # select the maximum q value
        #print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        # 这里是按批次计算的
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # (32,)一维=>(batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad() # reset the gradient to zero
        loss.backward()
        self.optimizer.step() # execute back propagation for one step
 
dqn = DQN()
max_i=0
for i_episode in range(400):
    # play 400 episodes of cartpole game
    s,_= env.reset()
    total_r = 0

    total_i=0
    while True:
        total_i+=1
        #最多坚持3000步就够了
        if(total_i>3000):
            break
        a = dqn.choose_action(s)

        s_, r, terminated, truncated, info = env.step(a)
        done=terminated or truncated

        # modify the reward based on the environment state
        x, x_dot, theta, theta_dot = s_# x:车的位置，x_dot:车的速度，theta:车的角度，theta_dot:车的角度速度
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        
        dqn.store_transition(s, a, r, s_)
        
        total_r += r
        # if the experience repaly buffer is filled, DQN begins to 交互式 learn or update
        # its parameters. 这样经验足够多，才能学习到更好的策略。
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                done_num+=1
        
        if done:
            if(total_i>max_i):
                max_i=total_i
                print(f'第{i_episode} episode,max_i:{max_i}')
            break
        s = s_  
    if i_episode%100==0:
        print(f'第{i_episode} episode')
        done_num=0
print("training done")
env.close()
os.system("pause")

env = gym.make("CartPole-v0", render_mode="human")
env = env.unwrapped
s, _ = env.reset()

total_r = 0
i_steps = 0
done = False

while not done:
    i_steps += 1
    #最多坚持3000步就够了
    if i_steps>3000:
        break

    # 推理模式（纯贪心）
    a = dqn.predict(s)
    s_, r, terminated, truncated, info = env.step(a)
    done = terminated or truncated

    # reward shaping
    x, x_dot, theta, theta_dot = s_

    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    r = r1 + r2

    total_r += r
    s = s_

print('steps: ', i_steps, ' | total_r: ', round(total_r, 2))
os.system("pause")
env.close()
