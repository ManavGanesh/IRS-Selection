import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Residual block with swish (SiLU) activation
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.SiLU()  # swish activation

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        return out + residual

# The Q-network with dual input branches
class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNet, self).__init__()
        # state_size is a list: [dim_branchA, dim_branchB]
        self.branchA_input = state_size[0]
        self.branchB_input = state_size[1]
        
        # Branch A for channel estimate features
        self.branchA_fc1 = nn.Linear(self.branchA_input, 128)
        self.branchA_resblocks = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.branchA_fc2 = nn.Linear(128, 128)
        
        # Branch B for IRS reflection pattern features
        self.branchB_fc1 = nn.Linear(self.branchB_input, 128)
        self.branchB_resblocks = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.branchB_fc2 = nn.Linear(128, 128)
        
        # Combined layers after concatenation
        self.combined_fc1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)
        
        self.activation = nn.SiLU()  # swish activation

    def forward(self, stateA, stateB):
        # stateA: (batch, dim_branchA)
        # stateB: (batch, dim_branchB)
        xA = self.activation(self.branchA_fc1(stateA))
        xA = self.branchA_resblocks(xA)
        xA = self.activation(self.branchA_fc2(xA))
        
        xB = self.activation(self.branchB_fc1(stateB))
        xB = self.branchB_resblocks(xB)
        xB = self.activation(self.branchB_fc2(xB))
        
        # Concatenate along feature dimension
        x = torch.cat((xA, xB), dim=1)  # shape: (batch, 256)
        x = self.activation(self.combined_fc1(x))  # shape: (batch, 256)
        q = self.out(x)                            # shape: (batch, action_size)
        return q

class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size  # list: [dim_branchA, dim_branchB]
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.device = device
        
        self.model = DQNNet(state_size, action_size).to(self.device)
        self.target_model = DQNNet(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        
    def act(self, state):
        # state is a tuple: (stateA, stateB), each a numpy array of shape (1, dim)
        if np.random.rand() <= self.epsilon:
            act_type = 'random'
            return random.randrange(self.action_size), act_type
        self.model.eval()
        with torch.no_grad():
            stateA = torch.tensor(state[0], dtype=torch.float32).to(self.device)
            stateB = torch.tensor(state[1], dtype=torch.float32).to(self.device)
            q_values = self.model(stateA, stateB)  # shape: (1, action_size)
        act_type = 'RL'
        action = torch.argmax(q_values, dim=1).item()
        return action, act_type
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0

        minibatch = random.sample(self.memory, batch_size)
        # Ensure the most recent experience is included
        minibatch[0] = self.memory[-1]
        
        stateA_batch = []
        stateB_batch = []
        next_stateA_batch = []
        next_stateB_batch = []
        actions = []
        rewards = []
        for state, action, reward, next_state in minibatch:
            stateA_batch.append(state[0])   # shape: (1, branchA_dim)
            stateB_batch.append(state[1])   # shape: (1, branchB_dim)
            next_stateA_batch.append(next_state[0])
            next_stateB_batch.append(next_state[1])
            actions.append(action)
            rewards.append(reward)
            
        # Convert lists to tensors
        stateA_batch = torch.tensor(np.array(stateA_batch), dtype=torch.float32).to(self.device).squeeze(1)
        stateB_batch = torch.tensor(np.array(stateB_batch), dtype=torch.float32).to(self.device).squeeze(1)
        next_stateA_batch = torch.tensor(np.array(next_stateA_batch), dtype=torch.float32).to(self.device).squeeze(1)
        next_stateB_batch = torch.tensor(np.array(next_stateB_batch), dtype=torch.float32).to(self.device).squeeze(1)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Forward pass to get current Q-values
        self.model.train()
        q_values = self.model(stateA_batch, stateB_batch)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values using double DQN logic
        with torch.no_grad():
            next_actions = self.model(next_stateA_batch, next_stateB_batch).argmax(dim=1)
            next_q_values = self.target_model(next_stateA_batch, next_stateB_batch)
            next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * next_q
            
        # Compute loss and update
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()
