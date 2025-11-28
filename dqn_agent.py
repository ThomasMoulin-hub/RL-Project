import torch
import torch.nn as nn
from gnn import BipartiteGNN
import numpy as np


class DQNAgent:
    def __init__(self, input_dim, hidden_dim, action_dim, global_dim):
        # Main Q-network: approximates Q(s,a) using the bipartite GNN
        self.model = BipartiteGNN(
            var_in_channels=2,
            con_in_channels=2,
            hidden_channels=hidden_dim,
            num_actions=action_dim,
            global_feat_size=global_dim
        )
        # Target Q-network: same architecture, used to compute stable TD targets
        self.target_model = BipartiteGNN(
            var_in_channels=2,
            con_in_channels=2,
            hidden_channels=hidden_dim,
            num_actions=action_dim,
            global_feat_size=global_dim
        )
        # Start with identical weights in both networks
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Optimizer and loss (Huber / SmoothL1 is standard in DQN)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # ε-greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.action_dim = action_dim

        # Discount factor and target network update schedule
        self.gamma = 0.99
        self.update_counter = 0
        self.target_update_freq = 50  # sync target net every 50 gradient updates

    def get_action(self, pyg_data, global_feats, greedy=False):
        """
        ε-greedy policy on top of Q(s,·):
        - if greedy=True: always use argmax Q(s,a)
        - else: with prob ε, pick a random action (exploration)
        """
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.action_dim)

        self.model.eval()
        with torch.no_grad():
            # Global features: numpy -> tensor, add batch dim
            global_tensor = torch.from_numpy(global_feats).float().unsqueeze(0)
            # All nodes belong to graph index 0
            batch_idx = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
            # Q(s,·) for the current state
            q_values = self.model(
                pyg_data.x,
                pyg_data.edge_index,
                batch_idx,
                global_tensor
            )
        # Return greedy action
        return torch.argmax(q_values).item()

    def update(self, batch_data):
        """
        One DQN update pass over all transitions in `batch_data`.
        Each element is (pyg, glob, act, rew, next_pyg, next_glob, done).
        """
        if not batch_data:
            return 0.0

        self.model.train()
        loss_total = 0.0

        for pyg, glob, act, rew, next_pyg, next_glob, done in batch_data:
            # Current state
            glob_t = torch.from_numpy(glob).float().unsqueeze(0)
            batch_idx = torch.zeros(pyg.num_nodes, dtype=torch.long)

            # Q(s,·) from current network
            q_values = self.model(pyg.x, pyg.edge_index, batch_idx, glob_t)
            current_q = q_values[0, act]  # Q(s,a) for the action taken

            # DQN target: r + γ max_a' Q_target(s',a') if not done
            target_q = rew
            if not done:
                with torch.no_grad():
                    next_glob_t = torch.from_numpy(next_glob).float().unsqueeze(0)
                    next_batch_idx = torch.zeros(next_pyg.num_nodes, dtype=torch.long)
                    next_q = self.target_model(
                        next_pyg.x,
                        next_pyg.edge_index,
                        next_batch_idx,
                        next_glob_t
                    )
                    target_q += self.gamma * torch.max(next_q).item()

            target_tensor = torch.tensor(target_q, dtype=torch.float)

            # Huber loss between predicted Q(s,a) and target
            loss = self.criterion(current_q, target_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to avoid exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_total += loss.item()

            # Periodically sync target network with main network
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        # ε decay after each update call
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return loss_total / len(batch_data)
