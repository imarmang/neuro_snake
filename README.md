# Neuro_Snake
![Neuro_Snake Demo](assets/demo.gif)


Neuro_Snake is an AI-enhanced implementation of the classic Snake game, using Deep-Q-Network Learning. The agent uses a neural network to learn optimal strategies for navigating the grid, collecting food, and avoiding collisions. With continuous training, the model improves its decision-making by minimizing loss and maximizing cumulative rewards. 

## Requirements

Install requirements using the following command:
```bash
pip install -r requirements.txt
```

## Run the Code
```bash
cd src/
python agent.py
```

## AI Training
The AI uses the DQN algorithm to learn how to play Snake. During training, it explores different actions in the game, such as moving in various directions, and observes the rewards obtained for actions like eating food or avoiding collisions. The AI's goal is to maximize its rewards by adjusting its actions based on the Q-values learned from the DQN, ultimately improving its ability to navigate the game board, grow longer, and survive as long as possible.

## Deep Q-Network (DQN) Learning
The training process of the Snake AI involves using the Deep Q-Network (DQN) algorithm, a powerful method in reinforcement learning. DQN combines traditional Q-learning with deep neural networks to approximate the Q-values for different state-action pairs.

                                                 Q(s, a) = r + γ * max​Q(s′, a′)

* The Linear_QNet class defines a simple feedforward neural network with one hidden layer. It maps input states to Q-values for each possible action.
* The QTrainer class handles training by:
    1. Predicting the Q-values for the current state using the network.
    2. Computing the target Q-values using the formula.
    3. Updating the network weights to minimize the loss between the predicted Q-values and the computed targets.
