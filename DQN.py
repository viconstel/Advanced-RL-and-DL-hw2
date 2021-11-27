import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy


class ReplayMemory():
    """Replay memory buffer."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, exptuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exptuple
        self.position = (self.position + 1) % self.capacity
       
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN:
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, batch_size=256, lr=3e-4, gamma=0.99):
        self.steps = 0
        self.buffer = ReplayMemory(10000)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma

        self.model = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

#       Target Q-network for stability
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = Adam(self.model.parameters(), lr)
    
    def act(self, state):
        self.model.eval()
        action = self.model(torch.FloatTensor([state]))
        return action.argmax().item()

    def get_empty_cells_indices(self, state):
        return np.where(state == 1)[0]
    
    def make_action(self, state, epsilon=0.1):
        state = np.array(list(map(int, state)))
        
        if random.random() < (1 - epsilon):
            action = self.act(state)
        else:
            action = random.choice(self.get_empty_cells_indices(state))
            
        return action

    def train_step(self, batch):
        state, action, next_state, reward, done = list(zip(*batch))
        state = torch.tensor(np.array(state, dtype=np.float32))
        action = torch.tensor(np.array(action, dtype=np.int64))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        reward = torch.tensor(np.array(reward, dtype=np.float32))
        done = torch.tensor(np.array(done, dtype=np.bool8))
        
        self.model.train()
        Q_values = self.model(state).gather(1, action.reshape(-1, 1)).squeeze()

        with torch.no_grad():
            Q_prime_values = self.target_model(next_state)

        target_values = torch.max(Q_prime_values, dim=1).values
        target_values[done] = 0.
        target_values = reward + self.gamma * target_values

        loss = F.mse_loss(Q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update(self, transition, steps_per_update=4, steps_per_target_update=400):
        self.buffer.store(transition)
        
        if self.steps % steps_per_update == 0:
            if len(self.buffer) < self.batch_size:
                return
    
            batch = self.buffer.sample(self.batch_size)
            self.train_step(batch)
        if self.steps % steps_per_target_update == 0:
            self.update_target_network()
        self.steps += 1
