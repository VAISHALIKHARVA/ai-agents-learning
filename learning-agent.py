import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

WIDTH, HEIGHT = 800, 800
GRID_SIZE = 5
CELL_SIZE = WIDTH // GRID_SIZE
FSP = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

#Action
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snack Q-Learning Agent - Small Grid")
font = pygame.font.SysFont(None, 24)

#Food class
class Food:
    def __init__(self):
        self.position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    def randomise(self):
        self.position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

class Snack:
    def __init__(self):
        self.position = [(2, 2)]
        self.direction = random.choice(ACTIONS)

    def move(self):
        self.direction = action
        head_x, head_y = self.position[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        new_head = (new_head[0] % GRID_SIZE, new_head[1] % GRID_SIZE)  # Wrap around
        self.position.insert(0, new_head)

    def grow(self):
        tail = self.position[-1]
        self.position.append(tail)

    def collision(self):
        head = self.position[0]
        return (
            self.position[0] in self.position[1:] or
            head[0] < 0 or head[0] >= GRID_SIZE or
            head[1] < 0 or head[1] >= GRID_SIZE
        )

# initialize q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.2 # exploration rate
num_episodes = 50

def get_state(snack):
    return snack.position[0]

def get_reward(snack, food):
    if snack.position[0] == food.position:
        return 10
    elif snack.collision():
        return -10
    else:
        return -1
    
# Main game loop
clock = pygame.time.Clock()

for episode in range(1, num_episodes + 1):
    snack = Snack()
    food = Food()
    total_reward = 0
    done = False

    while not done:
        clock.tick(FSP)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state = get_state(snack)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(ACTIONS)
        else:
            action_index = np.argmax(q_table[state[0], state[1]])

        action = ACTIONS[action_index]
        snack.move()
        reward = get_reward(snack, food)
        total_reward += reward

        new_state = get_state(snack)
        if 0 <= new_state[0] < GRID_SIZE and 0 <= new_state[1] < GRID_SIZE:
            q_table[state[0], state[1], ACTIONS.index(action)] += alpha * (reward + gamma * np.max(q_table[new_state[0], new_state[1]]) - q_table[state[0], state[1], ACTIONS.index(action)])

        if snack.position[0] == food.position:
            snack.grow()
            food.randomise()

        #draw everything
        win.fill(WHITE)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(win, BLACK, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        for pos in snack.position:
            pygame.draw.rect(win, GREEN, (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            label = font.render(f"{pos}", True, BLACK)
            win.blit(label, (pos[0] * CELL_SIZE + 5, pos[1] * CELL_SIZE + 5))

        pygame.draw.rect(win, RED, (food.position[0] * CELL_SIZE, food.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        label = font.render(f"{food.position}", True, BLACK)
        win.blit(label, (food.position[0] * CELL_SIZE + 5, food.position[1] * CELL_SIZE + 5))

        pygame.display.update()

        if snack.collision():
            done = True

    print(f"Episode {episode}:Finished With Total Reward: {total_reward}")