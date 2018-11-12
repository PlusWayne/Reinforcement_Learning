import numpy as np
import time
import sys
sys.path.append('game')
import wrapped_flappy_bird as fb

game_state = fb.GameState()
for i in range(100):
    action = np.zeros(2)
    action[0] = 1
    time.sleep(1)
    img, reward, terminal = game_state.frame_step(action)
    print((i,reward, terminal))