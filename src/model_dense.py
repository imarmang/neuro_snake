import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, optimizer, model_file_name='model.pth', optimizer_model_name='optimizer.pth'):
        # Create a directory to save the model and optimizer if it doesn't exist
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Save the model's weights
        model_file_path = os.path.join(model_folder_path, model_file_name)
        torch.save(self.state_dict(), model_file_path)

        # Save the optimizer's state
        optimizer_file_path = os.path.join(model_folder_path, optimizer_model_name)
        torch.save(optimizer.state_dict(), optimizer_file_path)

        print('New weights and the optimizers are saved.')

    def load(self, file_name='model.pth'):
        # Load a previously saved model if available
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            print(f"Model loaded successfully from {file_path}")

        else:
            print(f"No saved model found at {file_path}. Starting fresh.")


# Define a helper class to train the Q-learning agent
class QTrainer:
    def __init__(self, model, lr, gamma, file_name='optimizer.pth'):
        self.lr = lr
        self.gamma = gamma  # Discount factor: balances immediate and future rewards
        self.model = model
        self.file_name = file_name
        self.optimizer = self._initialize_optimizer()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()

    def _initialize_optimizer(self):
        # Check if a saved optimizer exists and load it
        model_folder_path = './model'
        file_path = os.path.join(model_folder_path, self.file_name)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            optimizer.load_state_dict(checkpoint)
            print(f"Optimizer loaded successfully from {file_path}")
        else:
            # Otherwise, create a new optimizer
            print(f"No saved optimizer found at {file_path}. Creating a new optimizer.")

        return optimizer

    def train_step(self, state, action, reward, next_state, done):
        # Convert input data into tensors
        state = torch.tensor(np.array(state, dtype=np.float32))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        action = torch.tensor(np.array(action, dtype=np.float32))
        reward = torch.tensor(np.array(reward, dtype=np.float32))

        # Add a batch dimension if the input is a single sample
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predict Q values with the current state
        pred = self.model(state)

        # Create a copy of predictions to update specific Q-values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # Update the predicted Q-value for the taken action
            target[idx][torch.argmax(action).item()] = Q_new

        # 2.Q_new =  r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        # Update the model weights using the optimizer
        self.optimizer.step()