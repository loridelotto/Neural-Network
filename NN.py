import numpy as np
inputs = np.array([1, 2, 3, 2.5])
weights = np.array([0.2, 0.8, -0.5, 1])
bias = 2

output = np.dot(inputs, weights) + bias
print(output)