import sys
import gym

def placeholder_controller(state):
    return 2 # go right

if len(sys.argv) < 2:
    print("mountaincar.py takes one argument: the output directory of openai gym monitor data")
    sys.exit(1)

env = gym.make("MountainCar-v0")
env.monitor.start(sys.argv[1], force=True)
state = env.reset()
for t in range(200):
    env.render(mode='rgb_array')
    action = placeholder_controller(state)
    state, reward, is_terminal, _ = env.step(action)
    if is_terminal:
        break
env.monitor.close()
