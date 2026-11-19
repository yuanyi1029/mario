import time
import random
import numpy as np 
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
        
        enemies = self.get_enemies()
        for enemy in enemies: 
            row, col = self.to_cell(enemy[0], enemy[1])
            if 0 <= row < 13 and 0 <= col < 16: 
                grid[row, col] = -1 

        mario = self.get_mario()
        row, col = self.to_cell(mario[0], mario[1]) 
        if 0 <= row < 13 and 0 <= col < 16: 
            grid[row, col] = 2

        return grid 
    
    def read(self, address): 
        return int(self.ram[address])

    def to_cell(self, level_x, level_y):
        camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)
        row = (level_y // 16) - 1 
        col = (level_x // 16) - (camera_left_x // 16)
        
        return row, col
    
    def get_blocks(self):
        blocks = np.zeros((Grid.ROWS, Grid.COLS), dtype=int)

        camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)

        for i in range(Grid.ROWS):
            for j in range(Grid.COLS):
                level_col = camera_left_x // 16 + j
                buffer_col = level_col % 32
                
                page = buffer_col // 16
                offset = buffer_col % 16        
                address = 0x500 + (page * 208) + ((i * 16) + offset)
                
                if self.read(address) != 0:
                    blocks[i, j] = 1

        return blocks

    def get_enemies(self): 
        enemies = []
        for i in range(5): 
            if self.read(0xF + i) == 1: 
                camera_left_x = self.read(0x71A) * 256 + self.read(0x71C)
                level_x = self.read(0x6E + i) * 256 + self.read(0x87 + i) 
                level_y = self.read(0xCF + i) 
                screen_x = level_x - camera_left_x

                enemies.append((level_x, level_y, screen_x))

        return enemies 

    def get_mario(self): 
        level_x = self.read(0x6D) * 256 + self.read(0x86)
        level_y = self.read(0xCE)
        screen_x = self.read(0x3AD)

        return (level_x, level_y, screen_x)

    
# print(env.buttons)
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
OPTIMAL_ACTIONS = {
    "RIGHT":                [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    "RIGHT_JUMP":           [0, 0, 0, 0, 0, 0, 0, 1, 1],
    "SPRINT_RIGHT_JUMP":    [1, 0, 0, 0, 0, 0, 0, 1, 1],
}

# # Initialise environment
# env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
# env = MaxAndSkipObservation(env, skip=4)
# env.reset()

# # Initialise variables 
# best_actions = []
# best_reward = -1 

# # Main loop 
# start_time = time.perf_counter()

# for i in range(40): 
#     print(f"Searching through iteration {i + 1}...")
#     env.reset()
    
#     counter = 0 
#     actions = []
#     reward = 0 
#     attempt_best_reward = 0 
#     initial_lives = None 
#     isEnd = False 

#     while not isEnd:
#         # Sample better actions 
#         # action = env.action_space.sample()
#         action = random.choice(list(OPTIMAL_ACTIONS.values()))
#         actions.append(action)

#         observation, reward, terminated, truncated, info = env.step(action)
#         counter += 1

#         if not initial_lives: 
#             initial_lives = info["lives"]
#         terminated = terminated or info["lives"] < initial_lives
#         reward = info["xscrollHi"] * 256 + info["xscrollLo"] 
#         if reward > attempt_best_reward: 
#             attempt_best_reward = reward
#             counter = 0 
#         isEnd = terminated or truncated 

#         if isEnd: 
#             break 

#         # Early stopping 
#         if counter > 300: 
#             break 

#     if reward > best_reward: 
#         best_actions = actions 
#         best_reward = reward 

# end_time = time.perf_counter()
# total_time = end_time - start_time

# print(f"Search complete! Time taken: {total_time:.2f}s")
# print(f"Best reward: {best_reward}")

# env.close() 

# # Replay
# replay_env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
# replay_env.reset()  

# viewer = FrameViewer() 
# for action in best_actions: 
#     for _ in range(4): 
#         observation, reward, terminated, truncated, info = replay_env.step(action) 

#         frame = replay_env.render()
#         viewer.show(frame)

# replay_env.close()