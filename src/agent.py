import os
import argparse
import torch
import random
import numpy as np
from config import BATCH_SIZE, TrainConfig
from game import SnakeGameAI, Direction, Point
from collections import deque
from model_dense import Linear_QNet, QTrainer
from utils_seed import seed_everything
from utils_logger import CSVLogger

from utils_plot import plot


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["dqn", "double_dqn"], default="dqn",
                   help="Which algorithm to run.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--run_name", type=str, default="default",
                   help="Subfolder name under runs/<algo>/ for this experiment.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="Torch device: auto uses cuda if available, else cpu.")
    p.add_argument("--eval", action="store_true", help="Run evaluation only (no training).")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes for evaluation.")
    p.add_argument("--no_render", action="store_true", help="Disable UI rendering during eval if supported.")

    return p.parse_args()


class Agent:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = self.cfg.gamma

        self.device = cfg.torch_device

        self.memory = deque(maxlen=cfg.max_memory)  # popleft(), when the memory passes 100_000
        self.model = Linear_QNet(cfg.input_size, cfg.hidden_size, cfg.output_size)  # Neural network for estimating Q-values
        self.model.to(self.device)

        self.target_model = Linear_QNet(cfg.input_size, cfg.hidden_size, cfg.output_size)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.total_steps = 0  # counts environment steps for target updates

        self.trainer = QTrainer(
            self.model, 
            lr=cfg.lr, 
            gamma=cfg.gamma,
            algo=cfg.algo,
            target_model=self.target_model, 
            model_dir=cfg.run_dir,
            device=self.device
        )

        os.makedirs(cfg.run_dir, exist_ok=True)
        self.model.load(model_dir=cfg.run_dir)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict(), map_location=self.device)

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
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)  # calling the forward function
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def evaluate(cfg: TrainConfig, episodes: int):
    # Create agent + game
    agent = Agent(cfg)
    game = SnakeGameAI(cfg.width, cfg.height, cfg.food_timeout, cfg.algo)

    # Force evaluation behavior
    agent.epsilon = -10**9  # ensures we never explore
    agent.model.eval()

    logger = CSVLogger(
        filepath=f"{cfg.run_dir}/eval.csv",
        fieldnames=["episode", "score", "frames_alive"]
    )

    scores = []
    frames = []

    for ep in range(1, episodes + 1):
        game.reset()
        done = False

        while not done:
            state = agent.get_state(game)

            # Greedy action only
            state0 = torch.tensor(state, dtype=torch.float, device=agent.device)
            with torch.no_grad():
                prediction = agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1

            reward, done, score = game.play_step(final_move)

        scores.append(score)
        frames_alive = game.frame_iteration
        frames.append(frames_alive)

        logger.log({"episode": ep, "score": score, "frames_alive": frames_alive})
        print(f"[EVAL {cfg.algo}/{cfg.run_name}/seed={cfg.seed}] ep={ep}/{episodes} score={score} frames={frames_alive}")

    # Summary
    scores_arr = np.array(scores, dtype=np.float32)
    frames_arr = np.array(frames, dtype=np.float32)

    print("\n===== EVAL SUMMARY =====")
    print(f"Run: {cfg.algo}/{cfg.run_name} (seed={cfg.seed})")
    print(f"Episodes: {episodes}")
    print(f"Score: mean={scores_arr.mean():.3f} std={scores_arr.std():.3f} min={scores_arr.min():.0f} max={scores_arr.max():.0f}")
    print(f"Frames: mean={frames_arr.mean():.1f} std={frames_arr.std():.1f} min={frames_arr.min():.0f} max={frames_arr.max():.0f}")
    print(f"Saved eval logs to: {cfg.run_dir}/eval.csv\n")


def train(cfg):
    logger = CSVLogger(
        filepath=f"{cfg.run_dir}/metrics.csv",
        fieldnames=["episode", "score", "mean_score", "mean100", "frames_alive", "epsilon", "total_steps", "record"]
    )

    plot_scores = []
    plot_mean_scores = []
    plot_survival_time = []
    total_score = 0
    record = 0
    
    agent = Agent(cfg)
    game = SnakeGameAI(cfg.width, cfg.height, cfg.food_timeout, cfg.algo)

    last_100_scores = deque(maxlen=100)

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        agent.total_steps += 1
        if agent.total_steps % cfg.target_update_steps == 0:
            agent.update_target()

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            survival_time = game.frame_iteration
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save the model if it achieves a new record score
            if score > record:
                record = score
                agent.model.save(agent.trainer.optimizer, model_dir=cfg.run_dir)

            # print(f'Game {agent.n_games}, Score {score}, Record: {record}, Survival Frames {survival_time}')
            total_score += score
            mean_score = total_score / agent.n_games

            last_100_scores.append(score)
            mean100 = sum(last_100_scores) / len(last_100_scores)

            logger.log({
                "episode": agent.n_games,
                "score": score,
                "mean_score": mean_score,
                "mean100": mean100,
                "frames_alive": survival_time,
                "epsilon": agent.epsilon,
                "total_steps": agent.total_steps,
                "record": record,
            })
            print(
                f"[{cfg.algo}/{cfg.run_name}/seed={cfg.seed}] "
                f"Game# {agent.n_games} score={score} record={record} mean_100={mean100:.2f} "
                f"survived_frames={survival_time} steps={agent.total_steps} eps={agent.epsilon}"
            )


            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            plot_survival_time.append(survival_time)

            # Update the plots live
            plot(plot_scores, plot_mean_scores, plot_survival_time)


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(algo=args.algo, seed=args.seed, run_name=args.run_name, device=args.device)

    seed_everything(cfg.seed)

    if args.eval:
        evaluate(cfg, episodes=args.episodes)
    else:
        train(cfg)

