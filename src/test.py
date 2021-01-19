import numpy as np

state = np.random.random((1, 10, 3))

print(state)

channel_state = state[0, :, 0]

print(channel_state)

pointer = np.zeros((1, 3))
print(1 in pointer)
