import time
import random
import gymnasium as gym
import stable_retro as retro
from mario.render import FrameViewer

# print(env.buttons)
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
OPTIMAL_ACTIONS = {
    "RIGHT":                [0, 0, 0, 0, 0, 0, 0, 1, 0], 
    "RIGHT_JUMP":           [0, 0, 0, 0, 0, 0, 0, 1, 1],
    "SPRINT_RIGHT_JUMP":    [1, 0, 0, 0, 0, 0, 0, 1, 1],
}

# Initialise environment
env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
env.reset()

# Initialise variables 
best_actions = []
best_reward = -1 

# Main loop 
start_time = time.perf_counter()

for i in range(40): 
    print(f"Searching through iteration {i + 1}...")
    env.reset()
    
    counter = 0 
    actions = []
    reward = 0 
    attempt_best_reward = 0 
    initial_lives = None 
    isEnd = False 

    while not isEnd:
        # Sample better actions 
        # action = env.action_space.sample()
        action = random.choice(list(OPTIMAL_ACTIONS.values()))
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

        if isEnd: 
            break 

        # Early stopping 
        if counter > 1200: 
            break 

    if reward > best_reward: 
        best_actions = actions 
        best_reward = reward 

end_time = time.perf_counter()
total_time = end_time - start_time

print(f"Search complete! Time taken: {total_time:.2f}s")
print(f"Best reward: {best_reward}")

# Replay
env.reset() 

viewer = FrameViewer() 
for action in best_actions: 
    observation, reward, terminated, truncated, info = env.step(action) 

    frame = env.render() 
    viewer.show(frame)

env.close() 