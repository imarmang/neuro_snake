import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


def _move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
    """Move any tensor values inside an optimizer state_dict to the given device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, optimizer, model_file_name='model.pth', optimizer_model_name='optimizer.pth', model_dir='./model'):
        # Create a directory to save the model and optimizer if it doesn't exist
        
        os.makedirs(model_dir, exist_ok=True)

        # Save the model's weights
        model_file_path = os.path.join(model_dir, model_file_name)
        torch.save(self.state_dict(), model_file_path)

        # Save the optimizer's state
        optimizer_file_path = os.path.join(model_dir, optimizer_model_name)
        torch.save(optimizer.state_dict(), optimizer_file_path)

        print(f"Saved model + optimizer to {model_dir}")

    def load(self, file_name='model.pth', model_dir='./model', map_location=None):
        # Load a previously saved model if available
        file_path = os.path.join(model_dir, file_name)

        if os.path.exists(file_path):
            state_dict = torch.load(file_path, map_location=map_location or "cpu", weights_only=True)
            self.load_state_dict(state_dict)
            print(f"Model loaded successfully from {file_path}")

        else:
            print(f"No saved model found at {file_path}. Starting fresh.")


# Define a helper class to train the Q-learning agent
class QTrainer:
    def __init__(self, model, lr, gamma, algo="dqn", target_model=None, file_name='optimizer.pth', model_dir='./model', device=None):
        self.lr = lr
        self.gamma = gamma  # Discount factor: balances immediate and future rewards
        self.model = model
        self.algo = algo
        self.target_model = target_model

        self.device = device or torch.device("cpu")

        self.file_name = file_name
        self.model_dir = model_dir
        self.optimizer = self._initialize_optimizer()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()

    def _initialize_optimizer(self):
        # Check if a saved optimizer exists and load it
        file_path = os.path.join(self.model_dir, self.file_name)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(checkpoint)
            _move_optimizer_state(optimizer, self.device)

            print(f"Optimizer loaded successfully from {file_path}")
        else:
            # Otherwise, create a new optimizer
            print(f"No saved optimizer found at {file_path}. Creating a new optimizer.")

        return optimizer

    def train_step(self, state, action, reward, next_state, done):
        # Convert input data into tensors
        state = torch.tensor(np.array(state, dtype=np.float32), device=self.device)
        next_state = torch.tensor(np.array(next_state, dtype=np.float32), device=self.device)
        action = torch.tensor(np.array(action, dtype=np.float32), device=self.device)
        reward = torch.tensor(np.array(reward, dtype=np.float32), device=self.device)

        # Add a batch dimension if the input is a single sample
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        # Predict Q values for current state
        pred = self.model(state)

        # Build targets
        target = pred.clone().detach()  # detach = target should not track gradients

        with torch.no_grad():
            for idx in range(len(done)):
                Q_new = reward[idx]

                if not done[idx]:
                    ns = next_state[idx].unsqueeze(0)  # (1, state_dim)

                    if self.algo == "double_dqn" and self.target_model is not None:
                        # Select action with online network
                        a_star = torch.argmax(self.model(ns), dim=1).item()

                        # Evaluate with target network
                        q_target_next = self.target_model(ns)[0, a_star]
                        Q_new = reward[idx] + self.gamma * q_target_next
                    else:
                        # Vanilla DQN
                        Q_new = reward[idx] + self.gamma * torch.max(self.model(ns))

                action_idx = torch.argmax(action[idx]).item()
                target[idx][action_idx] = Q_new

        # Optimize
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
