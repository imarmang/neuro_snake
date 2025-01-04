import torch
import random
import numpy as np
from config.constants import MAX_MEMORY, LEARNING_RATE, BATCH_SIZE, WIDTH, HEIGHT, FOOD_TIME_OUT
from game import SnakeGameAI, Direction, Point
from collections import deque
from model_dense import Linear_QNet, QTrainer
from helper import plot


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate, play around must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft(), when the memory passes 100_000
        self.model = Linear_QNet(11,  512, 3)  # Neural network for estimating Q-values
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

        self.model.load()

    # Extracts the current state of the game as input features for the model
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State representation: Danger, movement direction, and food location
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    # Stores the agent's experience in memory for replay
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY

    # Trains the model using experiences from the memory (batch sampling)
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        # Unpacks the batch into separate components
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Trains the model using the most recent experience (immediate learning)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Determines the next move based on the current state
    def get_action(self, state):
        # random moves: tradeoff between exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # calling the forward function
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    # scores_and_mean_scores = {'scores': [], 'mean_scores': []}
    plot_survival_time = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(WIDTH, HEIGHT, FOOD_TIME_OUT)
    while True:

        # get the current state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train the short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        survival_time = game.frame_iteration

        if done:
            # train long memory, plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save the model if it achieves a new record score
            if score > record:
                record = score
                agent.model.save(agent.trainer.optimizer)

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Survival Time', survival_time)

            total_score += score
            mean_score = total_score / agent.n_games

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot_survival_time.append(survival_time)

            # Update the plots live
            plot(plot_scores, plot_mean_scores, plot_survival_time)


if __name__ == '__main__':
    train()
