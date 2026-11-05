import gymnasium as gym
import stable_retro as retro
from mario.render import FrameViewer

# uv run python -m stable_retro.examples.interactive --game SuperMarioBros-Nes-v0

# Initialise environment
env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
env.reset()

# env.unwrapped.data.set_value("time", 20)
viewer = FrameViewer()

counter = 0 
reward = 0 
initial_lives = None 
isEnd = False 

while not isEnd:    
    # print(env.buttons)
    # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    
    # action = env.action_space.sample()
    if counter % 2 == 0: 
        action = [0, 0, 0, 0, 0, 0, 0, 1, 1]
    else:
        action = [0, 0, 0, 0, 0, 0, 0, 1, 0]

    counter += 1 

    observation, reward, terminated, truncated, info = env.step(action)

    if not initial_lives: 
        initial_lives = info["lives"]
    terminated = terminated or info["lives"] < initial_lives 
    reward = info["xscrollHi"] * 256 + info["xscrollLo"] 
    isEnd = terminated or truncated 
    
    frame = env.render()
    viewer.show(frame)

    if isEnd:
        print(f"Reward: {reward}, In-game score: {info["score"]}")
        reward = 0 
        # env.reset()
        env.close() 