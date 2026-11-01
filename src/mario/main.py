import gymnasium as gym
import stable_retro as retro

from mario.render import FrameViewer

# uv run python -m stable_retro.examples.interactive --game SuperMarioBros-Nes-v0

# Initialise environment
env = retro.make(game="SuperMarioBros-Nes-v0", render_mode="rgb_array")
observation, info = env.reset()

viewer = FrameViewer()

while viewer.is_open:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    frame = env.render()
    viewer.show(frame)
    
    if terminated or truncated:
        env.reset()

env.close()