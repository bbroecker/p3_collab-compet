import time

from pynput.keyboard import Key, Listener
from unityagents import UnityEnvironment

from madddpg_utils import EnvironmentWrapper

action = -1.

flip = True
user = False


def on_press(key):

    print('{0} pressed'.format(
        key))



def on_release(key):

    global action, user
    user = True
    if key == Key.up:
        # action = 0
        action = -1.
    elif key == Key.down:
        # action = 1
        action = -0.5
    elif key == Key.left:
        action = 0.0
        # action = 0.33
    elif key == Key.right:
        action = 0.5
        # action = 1
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released

if __name__ == '__main__':
    l = Listener(on_release=on_release, on_press=on_press)
    l.start()
    env = UnityEnvironment(file_name='environments/Soccer_Linux/Soccer.x86_64')
    env = EnvironmentWrapper(env, discrete_actions=True, skip_frames=0)
    env.reset(train_mode=False)
    while True:
        # if not user:
        #     action = -1. if flip else -0.5
        #     flip = not flip
        if not user:
            continue

        actions = [0, 0, 0, action]
        env.step(actions)
        user = False
