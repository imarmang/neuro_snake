import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
from config.constants import WHITE, RED, BLUE1, BLACK, BLOCK_SIZE, SPEED, GREEN

pygame.init()
font = pygame.font.Font('../assets/arial.ttf', 25)


# Enumeration to represent the four possible directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


class SnakeGameAI:
    def __init__(self, WIDTH, HEIGHT, FOOD_TIME_OUT):
        self.w = WIDTH
        self.h = HEIGHT
        self.food_timeout = FOOD_TIME_OUT
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Resets the game to its initial state.
        """
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.time_since_last_food = 0
        
    def _place_food(self):
        """
        Places a food item at a random location, ensuring it doesn't overlap with the snake.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        # Ensure food is not placed on the snake
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0  # if the snake eats food: +10, -10 if it doesn't, 0 if else
        game_over = False
        if (self.is_collision() or self.frame_iteration > 100*len(self.snake)
                or self.time_since_last_food > self.food_timeout*SPEED/2):
            game_over = True
            reward = -10

            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.time_since_last_food = 0  # Reset the timer when food is eaten
        else:
            self.snake.pop()
            self.time_since_last_food += 1  # Reset the timer when food is eaten

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """
        Checks for collisions with the wall or itself.
        """
        # hits boundary
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False

    def _update_ui(self):
        """
        Updates the game UI by drawing the snake, food, and score.
        """
        self.display.fill(BLACK)  # Or your custom background

        # Draw the snake
        pygame.draw.rect(self.display, RED,
                         pygame.Rect(self.snake[0].x, self.snake[0].y, BLOCK_SIZE, BLOCK_SIZE))  # Head
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  # Body

        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Display score
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()

    def _move(self, action):
        """
        Updates the snake's direction and position based on the action taken.
        """
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Determine the new direction
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_direction = clock_wise[next_idx]  # right change r -> d -> l -> u
        else:
            next_idx = (idx - 1) % 4
            new_direction = clock_wise[next_idx]  # left change r -> u -> l -> d

        self.direction = new_direction

        # Update the head position based on the new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

