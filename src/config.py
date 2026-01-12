from dataclasses import dataclass
from enum import Enum
import torch


# ---------- Game / UI constants ----------
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
GREEN = (0, 180, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

# Board size
WIDTH = 640
HEIGHT = 480

# ---------- RL hyperparameters ----------
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 1e-3

# Neural Network Size
INPUT_SIZE = 11
HIDDEN_SIZE = 512
OUTPUT_SIZE = 3

# Respawn food if snake hasn't eaten N times, but should be # Respawn food if snake hasn't eaten in N frames
FOOD_TIME_OUT = 5


@dataclass(frozen=True)
class TrainConfig:
    """
    Central training config.
    We'll extend this soon with algo="dqn" vs "double_dqn".
    """
    algo: str = "dqn"  # "dqn" | "double_dqn"
    gamma: float = 0.9  # discount rate, play around must be smaller than 1

    max_memory: int = MAX_MEMORY
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    
    input_size: int = INPUT_SIZE
    hidden_size: int = HIDDEN_SIZE
    output_size: int = OUTPUT_SIZE

    width: int = WIDTH
    height: int = HEIGHT
    food_timeout: int = FOOD_TIME_OUT

    base_run_dir = "runs"

    target_update_steps: int = 1000   # hard update every N env steps
    
    seed: int = 42
    run_name: str = "default"

    device: str = "auto"  # "auto" | "cpu" | "cuda"

    @property
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def run_dir(self):
        return f"{self.base_run_dir}/{self.algo}/{self.run_name}"
    

# Enumeration to represent the four possible directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4