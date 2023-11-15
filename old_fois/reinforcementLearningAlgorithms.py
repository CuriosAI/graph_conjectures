import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        n = input_dims[0]
        self.fc1 = nn.Linear(n, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(num_features=fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims + n, self.fc2_dims)
        self.bn2 = nn.BatchNorm1d(num_features=fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims+ n, self.fc2_dims)
        self.bn3 = nn.BatchNorm1d(num_features=fc2_dims)
        self.fc4 = nn.Linear(self.fc2_dims+ n, self.fc2_dims)
        self.bn4 = nn.BatchNorm1d(num_features=fc2_dims)
        self.fc5 = nn.Linear(self.fc2_dims+ n, self.fc2_dims)
        self.bn5 = nn.BatchNorm1d(num_features=fc2_dims)
        self.fc6 = nn.Linear(self.fc2_dims+ n, self.n_actions)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
        self.loss = nn.SmoothL1Loss()
        self.softmax = nn.Softmax()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, mode="train"):
        if mode == "eval":
            self.eval()
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(T.cat([x,state],1))))
        x = F.relu(self.bn3(self.fc3(T.cat([x,state],1))))
        x = F.relu(self.bn4(self.fc4(T.cat([x,state],1))))
        x = F.relu(self.bn5(self.fc5(T.cat([x,state],1))))
        actions = self.softmax(self.fc6(T.cat([x,state],1)))
        if mode == "eval":
            self.train()
        return actions

class Agent():
    def __init__(self, Q_online, Q_target, gamma, epsilon, batch_size, 
                 max_mem_size=10000, eps_end=0.001, eps_dec=5e-4, tau=0.0001, double=True, soft_update=True):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.Q_online = Q_online
        self.Q_target = Q_target
        self.deterministic = False
        self.Q_target.load_state_dict(self.Q_online.state_dict())
        self.action_space = [i for i in range(self.Q_online.n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.steps_cntr = 0
        self.tau = tau
        self.double = double
        self.soft_update = soft_update

        self.state_memory = np.zeros((self.mem_size, *self.Q_online.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.Q_online.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def choose_action(self, observation):
        observation = T.tensor(observation, dtype=T.float32, device=self.Q_online.device).unsqueeze(0)
        if self.deterministic or np.random.random() > self.epsilon:
            with T.no_grad():
                actions = self.Q_online.forward(observation,"eval")
                action = actions.max(1)[1].item()
                return action
        else:
            return T.tensor([[env.action_space.sample()]], device=self.Q_online.device, dtype=T.long).item()
    
    def learn(self):
        if self.mem_cntr < 3*self.batch_size:
            return
        
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size)
        
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_online.device)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_online.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_online.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_online.device)
        action_batch = T.tensor(self.action_memory[batch], dtype=T.int64).to(self.Q_online.device)
        
        qq = self.Q_online(state_batch).gather(1,action_batch.view(-1,1))
        if self.double:
            action = self.Q_online(new_state_batch).max(1)[1]
            with T.no_grad():
                q_next = self.Q_target(new_state_batch)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * q_next.gather(1,action.view(-1,1)).flatten()
        else:
            with T.no_grad():
                q_next = self.Q_target(new_state_batch).max(1)[0]
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * q_next

        
        loss = self.Q_online.loss(qq, q_target.unsqueeze(1)).to(self.Q_online.device)
        self.Q_online.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(self.Q_online.parameters(), 100)
        self.Q_online.optimizer.step()
        
        if self.soft_update:
            online_dict = self.Q_online.state_dict()
            target_dict = self.Q_target.state_dict()
            
            for key in target_dict:
                target_dict[key] = online_dict[key]*self.tau + target_dict[key]*(1-self.tau)
            self.Q_target.load_state_dict(target_dict)
        else:
            self.steps_cntr += 1
            if self.steps_cntr >= 1/self.tau:
                self.Q_target.load_state_dict(self.Q_online.state_dict())
                self.steps_cntr = 0
        if not(self.deterministic):
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

import gymnasium as gym
from environments import GraphEdgeFlipEnvironment
import matplotlib.pyplot as plt
import networkx as nx
import random
if __name__ == '__main__':
    seed = 41
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    nnodes = 5
    #env = gym.make("LunarLander-v2",render_mode="human")
    #env = gym.make("LunarLander-v2")
    #env = gym.make("CartPole-v1")
    env = GraphEdgeFlipEnvironment(size=nnodes)
    state, info = env.reset()
    lr = 1e-4
    # Q_online = DeeperQNetwork(lr=lr, n_actions=env.action_space.n,
    #                                input_dims=[len(state)])
    # Q_target= DeeperQNetwork(lr=lr, n_actions=env.action_space.n,
    #                                input_dims=[len(state)])
    
    Q_online = DeepQNetwork(lr=lr, n_actions=env.action_space.n,fc1_dims=256,fc2_dims=256,
                                   input_dims=[len(state)])
    Q_target= DeepQNetwork(lr=lr, n_actions=env.action_space.n,fc1_dims=256,fc2_dims=256,
                                   input_dims=[len(state)])
    agent = Agent(Q_online=Q_online,Q_target=Q_target, gamma=1, epsilon=0.9,
                  batch_size=1024, eps_end = 0.1,eps_dec=0.99,tau=0.0001, double=True, soft_update=True,max_mem_size=100*1000)
    
    scores, eps_history = [], []
    n_games = 5000
    
    for i in range(n_games):
        score = 0
        done = False
        observation, _ = env.reset(seed)
        duration = 0
        agent.deterministic = False and i > 500 and (i % 2 > 0)
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            done = done or info
            score += reward
            diff = 0
            diff = env.value() - np.sqrt(nnodes-1)-1
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
            duration += 1
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[-50:])
        
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %2f' % agent.epsilon, 'duration %.2f' % duration, 'diff %.2f' %diff)
        x = [i + 1 for i in range(n_games)]
        filename = 'lunar_lander_2020.png'
        