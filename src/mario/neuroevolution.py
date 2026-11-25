import os
import time
import copy 
import random
import torch
import stable_retro as retro
import matplotlib.pyplot as plt

from tqdm import tqdm
from gymnasium.wrappers import MaxAndSkipObservation
from mario.render import FrameViewer

GAME = "SuperMarioBros-Nes-v0"

# Button order used by stable-retro (env.unwrapped.buttons):
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# Actions copied from COMPLEX_MOVEMENT in gym-super-mario-bros
# https://github.com/Kautenja/gym-super-mario-bros
ACTIONS = {
    "NONE":                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "RIGHT":                [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "RIGHT_JUMP":           [0, 0, 0, 0, 0, 0, 0, 1, 1],
    "RIGHT_SPRINT":         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    "RIGHT_JUMP_SPRINT":    [1, 0, 0, 0, 0, 0, 0, 1, 1],
    "JUMP":                 [0, 0, 0, 0, 0, 0, 0, 0, 1],
    "LEFT":                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "LEFT_JUMP":            [0, 0, 0, 0, 0, 0, 1, 0, 1],
    "LEFT_SPRINT":          [1, 0, 0, 0, 0, 0, 1, 0, 0],
    "LEFT_JUMP_SPRINT":     [1, 0, 0, 0, 0, 0, 1, 0, 1],
    "DOWN":                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "UP":                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
}
ACTION_BUTTONS = list(ACTIONS.values())
ACTION_NAMES = list(ACTIONS.keys())

GENERATIONS = 1000
POPULATION_SIZE = 50
PARENT_SIZE = 10
# Probability of mutation for each weight / bias in a child  
MUTATION_RATE = 0.05 
# Standard deviation of Gaussian noise applied to mutated weights 
MUTATION_SCALE = 0.2 

class Grid: 

    ROWS = 13
    COLS = 16

    def __init__(self): 
        self.ram = None 

    def build(self, ram): 
        self.ram = ram  

        grid = self.get_blocks()

        # Add enemies to the grid
        enemies = self.get_enemies()
        for enemy in enemies: 
            row, col = self.to_cell(enemy[0], enemy[1])
            if 0 <= row < Grid.ROWS and 0 <= col < Grid.COLS: 
                grid[row, col] = -1 

        # Add mario to the grid 
        mario = self.get_mario()
        row, col = self.to_cell(mario[0], mario[1]) 
        if 0 <= row < Grid.ROWS and 0 <= col < Grid.COLS: 
            grid[row, col] = 2

        return grid 
    
    def read(self, address): 
        return int(self.ram[address])

    def to_cell(self, level_x, level_y):
        # Pixel value of the camera's left edge
        camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)
        
        row = (level_y // 16) - 1 
        col = (level_x // 16) - (camera_left_x // 16)
        
        return row, col
    
    def get_blocks(self):
        blocks = torch.zeros((Grid.ROWS, Grid.COLS))

        # Pixel value of the camera's left edge
        camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)

        for i in range(Grid.ROWS):
            for j in range(Grid.COLS):
                # Column in the level (0 - n)
                level_col = camera_left_x // 16 + j
                # Column number in RAM (0 - 31)
                buffer_col = level_col % 32

                # Page number (0 - 1) and offset (0 - 15) of the column 
                page = buffer_col // 16
                offset = buffer_col % 16

                # Address of the specific tile
                address = 0x500 + (page * 208) + ((i * 16) + offset)
                
                if self.read(address) != 0:
                    blocks[i, j] = 1

        return blocks

    def get_enemies(self): 
        enemies = []
        for i in range(5): 
            if self.read(0xF + i) == 1: 
                # Pixel value of the camera's left edge
                camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)

                # Pixel value of enemy's x and y position relative to the game
                level_x = self.read(0x6E + i) * 256 + self.read(0x87 + i) 
                level_y = self.read(0xCF + i) 

                # Pixel value of enemy's x position relative to the camera
                screen_x = level_x - camera_left_x

                enemies.append((level_x, level_y, screen_x))

        return enemies 

    def get_mario(self): 
        # Pixel value of mario's x and y position relative to the game
        level_x = self.read(0x6D) * 256 + self.read(0x86)
        level_y = self.read(0xCE)

        # Pixel value of mario's x position relative to the camera
        screen_x = self.read(0x3AD)

        return (level_x, level_y, screen_x)

class Network: 

    def __init__(self): 
        self.w1 = torch.randn(208, 32)
        self.b1 = torch.zeros(32)
        self.w2 = torch.randn(32, 16)
        self.b2 = torch.zeros(16)
        self.w3 = torch.randn(16, 12)
        self.b3 = torch.zeros(12)

    def forward(self, x):
        x = x.flatten() 
        x = torch.relu(x @ self.w1 + self.b1)
        x = torch.relu(x @ self.w2 + self.b2)
        x = x @ self.w3 + self.b3
        return x  

    def mutate(self, rate, scale): 
        for parameter in [
            self.w1, self.b1, 
            self.w2, self.b2, 
            self.w3, self.b3, 
        ]: 
            mask = torch.rand_like(parameter) < rate
            noise = torch.randn_like(parameter) * scale
            parameter += + noise * mask 

    def save(self, path):
        state = {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
        }
        torch.save(state, path)

    def load(self, path): 
        state = torch.load(path) 
        self.w1, self.b1 = state["w1"], state["b1"]
        self.w2, self.b2 = state["w2"], state["b2"]
        self.w3, self.b3 = state["w3"], state["b3"]

def evaluate_model(env, model):
    env.reset() 

    ram = env.unwrapped.get_ram()
    grid = Grid() 

    initial_info = env.unwrapped.data.lookup_all() 
    initial_timer = initial_info["time"] 
    initial_lives = initial_info["lives"] 

    actions = []
    counter = 0 
    fitness = 0 
    max_fitness = 0 
    max_distance = 0 
    is_end = False

    while not is_end:
        state = grid.build(ram) 
        logits = model.forward(state) 
        action = ACTION_BUTTONS[torch.argmax(logits)]
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        ram = env.unwrapped.get_ram()
        counter += 1 

        timer = info["time"]
        lives = info["lives"]
        distance = info["xscrollHi"] * 256 + info["xscrollLo"] 

        is_dead = lives < initial_lives
        is_complete = int(ram[0x001D]) == 3 
        is_end = terminated or truncated or is_dead or is_complete

        if distance > max_distance or is_complete: 
            fitness = calculate_fitness(distance, timer, initial_timer, is_complete)
            max_fitness = max(max_fitness, fitness)
            max_distance = distance
            counter = 0  
        
        if counter > 300: 
            break 

    return (max_fitness, actions)

def calculate_fitness(distance, timer, initial_timer, is_complete):
    return (distance * ((timer + 1) / (initial_timer + 1))) * (2 if is_complete else 1)

def train(): 
    # Initialise training folder 
    run_dir = f"runs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(run_dir)

    # Initialise environment
    env = retro.make(game=GAME, render_mode="rgb_array")
    env = MaxAndSkipObservation(env, skip=4)
    env.reset()

    population = [Network() for _ in range(POPULATION_SIZE)]
    history = {"mean": [], "max": []}

    for _ in tqdm(range(GENERATIONS), desc="Generation"): 
        # Evaluation 
        results = []
        for model in population: 
            fitness, _ = evaluate_model(env, model) 
            results.append((model, fitness))

        # Ranking 
        results = sorted(results, key=lambda x : x[1], reverse=True)

        mean_fitness = sum(fitness for _, fitness in results) / len(results)
        max_fitness = results[0][1]

        history["mean"].append(mean_fitness)
        history["max"].append(max_fitness)
        
        # Selection 
        parents = results[:PARENT_SIZE]
        elite, _ = parents[0] 

        # Crossover (uniform crossover / point crossover)
        # Skipped for now 

        # Mutation 
        population = [elite]

        for _ in range(POPULATION_SIZE - 1):
            model, _ = random.choice(parents)
            child = copy.deepcopy(model)
            child.mutate(MUTATION_RATE, MUTATION_SCALE) 
            population.append(child)

    elite, _ = results[0]
    elite.save(f"{run_dir}/model.pth")

    plt.plot(history["mean"], label="mean")
    plt.plot(history["max"], label="max")
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(f"{run_dir}/fitness.png")
    plt.show()

    env.close()

def play(): 
    # Initialise environment
    env = retro.make(game=GAME, render_mode="rgb_array")
    env = MaxAndSkipObservation(env, skip=4)
    env.reset()

    model = Network() 
    model.load("runs/2026-11-24_09-18-42/model.pth")

    # Main loop
    start_time = time.perf_counter()
    fitness, actions = evaluate_model(env, model) 
    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"Evaluation complete! Time taken: {total_time:.2f}s")
    print(f"Fitness: {fitness}")

    env.close() 

    # Replay
    replay_env = retro.make(game=GAME, render_mode="rgb_array")
    replay_env.reset()  

    viewer = FrameViewer() 
    for action in actions:
        for _ in range(4): 
            observation, reward, terminated, truncated, info = replay_env.step(action) 

            frame = replay_env.render()
            viewer.show(frame)

    replay_env.close()

if __name__ == "__main__": 
    # train()
    play()