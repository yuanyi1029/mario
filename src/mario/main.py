import gymnasium as gym 
import stable_retro as retro  

# uv run python -m stable_retro.examples.interactive --game SuperMarioBros-Nes-v0  

# Initialise environment 
# env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="human")
observation, info = env.reset() 

while True: 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    env.render()

    if terminated or truncated:
        env.reset()
