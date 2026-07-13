import numpy as np
import nnfs
from nnfs.datasets import spiral_data  
nnfs.init()
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs , n_neurons)             # weights matrix defined as inputs x neurons initialized with random numbers from normal distribution mu = 0 and sigma = 1
        self.bias = np.zeros((1, n_neurons))                                # bias is a column vector
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        
# Create dataset
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
dense1.forward(X)

print(dense1.output[:5])


