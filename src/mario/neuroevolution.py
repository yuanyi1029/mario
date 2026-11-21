import time
import torch
import stable_retro as retro

from gymnasium.wrappers import MaxAndSkipObservation
from mario.render import FrameViewer

class Grid: 

    ROWS = 13
    COLS = 16

    def __init__(self): 
        self.ram = None 

    def build(self, ram): 
        self.ram = ram  

        grid = self.get_blocks()

        # add enemies to the grid
        enemies = self.get_enemies()
        for enemy in enemies: 
            row, col = self.to_cell(enemy[0], enemy[1])
            if 0 <= row < Grid.ROWS and 0 <= col < Grid.COLS: 
                grid[row, col] = -1 

        # add mario to the grid 
        mario = self.get_mario()
        row, col = self.to_cell(mario[0], mario[1]) 
        if 0 <= row < Grid.ROWS and 0 <= col < Grid.COLS: 
            grid[row, col] = 2

        return grid 
    
    def read(self, address): 
        return int(self.ram[address])

    def to_cell(self, level_x, level_y):
        # pixel value of the camera's left edge
        camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)
        
        row = (level_y // 16) - 1 
        col = (level_x // 16) - (camera_left_x // 16)
        
        return row, col
    
    def get_blocks(self):
        blocks = torch.zeros((Grid.ROWS, Grid.COLS))

        # pixel value of the camera's left edge
        camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)

        for i in range(Grid.ROWS):
            for j in range(Grid.COLS):
                # column in the level (0 - n)
                level_col = camera_left_x // 16 + j
                # column number in RAM (0 - 31)
                buffer_col = level_col % 32

                # page number (0 - 1) and offset (0 - 15) of the column 
                page = buffer_col // 16
                offset = buffer_col % 16

                # address of the specific tile
                address = 0x500 + (page * 208) + ((i * 16) + offset)
                
                if self.read(address) != 0:
                    blocks[i, j] = 1

        return blocks

    def get_enemies(self): 
        enemies = []
        for i in range(5): 
            if self.read(0xF + i) == 1: 
                # pixel value of the camera's left edge
                camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)

                # pixel value of enemy's x and y position relative to the game
                level_x = self.read(0x6E + i) * 256 + self.read(0x87 + i) 
                level_y = self.read(0xCF + i) 

                # pixel value of enemy's x position relative to the camera
                screen_x = level_x - camera_left_x

                enemies.append((level_x, level_y, screen_x))

        return enemies 

    def get_mario(self): 
        # pixel value of mario's x and y position relative to the game
        level_x = self.read(0x6D) * 256 + self.read(0x86)
        level_y = self.read(0xCE)

        # pixel value of mario's x position relative to the camera
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

# button order used by stable-retro (env.unwrapped.buttons):
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
# actions copied from COMPLEX_MOVEMENT in gym-super-mario-bros
# https://github.com/Kautenja/gym-super-mario-bros
OPTIMAL_ACTIONS = {
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

ACTION_BUTTONS = list(OPTIMAL_ACTIONS.values())
ACTION_NAMES = list(OPTIMAL_ACTIONS.keys())

# Initialise environment
env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
env = MaxAndSkipObservation(env, skip=4)
env.reset()

grid = Grid()
network = Network() 

# Initialise variables 
counter = 0 
actions = []
attempt_best_reward = 0 
initial_lives = None 
isEnd = False 

# Main loop 
start_time = time.perf_counter()

while not isEnd: 
    state = grid.build(env.unwrapped.get_ram())
    logits = network.forward(state)
    action = ACTION_BUTTONS[torch.argmax(logits)]
    actions.append(action)

    observation, reward, terminated, truncated, info = env.step(action)
    counter += 1 

    if not initial_lives: 
        initial_lives = info["lives"]
    terminated = terminated or info["lives"] < initial_lives
    reward = info["xscrollHi"] * 256 + info["xscrollLo"] 
    if reward > attempt_best_reward: 
        attempt_best_reward = reward
        counter = 0
    isEnd = terminated or truncated

    # Early stopping 
    if counter > 300: 
        break 

end_time = time.perf_counter()
total_time = end_time - start_time

print(f"Evaluation complete! Time taken: {total_time:.2f}s")
print(f"Reward: {attempt_best_reward}")

env.close() 

# Replay
replay_env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
replay_env.reset()  

viewer = FrameViewer() 
for action in actions:
    for _ in range(4): 
        observation, reward, terminated, truncated, info = replay_env.step(action) 

        frame = replay_env.render()
        viewer.show(frame)

replay_env.close()