import gymnasium as gym 
import stable_retro as retro  
from stable_retro.examples.interactive import Interactive
from stable_retro.enums import State

GAME = "SuperMarioBros-Nes-v0"

class MarioInteractive(Interactive): 
    def __init__(self):
        env = retro.make(
            game=GAME,
            state=State.DEFAULT, 
            scenario=None,
            record=False,
            render_mode="rgb_array"
        )
        self._buttons = env.buttons
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4 / 3)

    def get_image(self, _obs, env): 
        return env.render() 

    def keys_to_act(self, keys):
        inputs = {
            None: False,
            "A": "SPACE" in keys,
            "B": "LSHIFT" in keys,
            "UP": "W" in keys,
            "DOWN": "S" in keys,
            "LEFT": "A" in keys,
            "RIGHT": "D" in keys,
            "SELECT": "TAB" in keys,
            "START": "ENTER" in keys,
        }
        return [inputs[b] for b in self._buttons]
        
def main(): 
    mi = MarioInteractive()
    mi.run()

if __name__ == "__main__": 
    main() 