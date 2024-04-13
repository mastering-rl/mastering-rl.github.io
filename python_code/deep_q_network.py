import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hashlib

from qfunction import QFunction


class DQN(nn.Module, QFunction):

    
    def __init__(self, state_space, action_space, hidden_dim=128, alpha=1e-4):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_space, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_space)
        self.optimiser = optim.AdamW(self.parameters(), lr=alpha, amsgrad=True)

    """ A forward pass through the network """

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def get_q_values(self, states, actions):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        q_values = self.forward(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        return q_values.squeeze(1).tolist()

    def get_q_value(self, state, action):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        q_values = self.forward(state_tensor)
        q_value = q_values[action]
        return q_value.item()

    def get_max_q_values(self, states):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            max_q_values = self(states_tensor).max(1).values
            return max_q_values.tolist()

    def get_max_q(self, state, actions):
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            return (self(state_tensor).max(1).indices.view(1, 1).item(), None)

    def update(self, states, actions, deltas):

        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        deltas_tensor = torch.tensor(deltas, dtype=torch.float32, requires_grad=True)

        # Compute Q-values for current states
        current_q_values = self.forward(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        )

        # Calculate the loss
        loss = nn.functional.smooth_l1_loss(
            current_q_values,
            (current_q_values.clone().detach().squeeze(1) + deltas_tensor).unsqueeze(1),
        )

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimiser.step()

    def soft_update(self, policy_qfunction, tau=0.005):
        target_dict = self.state_dict()
        policy_dict = policy_qfunction.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.load_state_dict(target_dict)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
