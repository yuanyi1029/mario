import gymnasium as gym
import stable_retro as retro
from mario.render import FrameViewer

# Initialise environment
env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
env.reset()

# Initialise variables 
best_actions = []
best_reward = -1 

# Main loop 
for i in range(20): 
    print(f"Searching through iteration {i + 1}...")
    env.reset()
    
    actions = []
    reward = 0 
    initial_lives = None 
    isEnd = False 

    while not isEnd: 
        action = env.action_space.sample()
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        if not initial_lives: 
            initial_lives = info["lives"]
        terminated = terminated or info["lives"] < initial_lives
        reward = info["xscrollHi"] * 256 + info["xscrollLo"] 
        isEnd = terminated or truncated 

        if isEnd: 
            break 

    if reward > best_reward: 
        best_actions = actions 
        best_reward = reward 

print("Search complete!")
print(f"Best reward: {best_reward}")

# Replay
env.reset() 

viewer = FrameViewer() 
for action in best_actions: 
    observation, reward, terminated, truncated, info = env.step(action) 

    frame = env.render() 
    viewer.show(frame)

env.close() 