import numpy as np
import nnfs
from nnfs.datasets import spiral_data  
nnfs.init()
import matplotlib.pyplot as plt

#Dense Layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs , n_neurons)             # weights matrix defined as inputs x neurons initialized with random numbers from normal distribution mu = 0 and sigma = 1
        self.bias = np.zeros((1, n_neurons))                                # bias is a column vector
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        
#ReLU activation function
class activation_reLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#SoftMax activation function
class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #unnormalized probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)
# Create ReLU activation
activation1 = activation_reLU()
# Create second Dense layer with 3 input features and 3 output values
dense2 = Layer_Dense(3,3)
# Create Softmax activation
activation2 = activation_softmax()

# Make a forward pass of our training data through the first layer
dense1.forward(X)
# Make a forward pass through the first activation function
activation1.forward(dense1.output)

# Make a forward pass through the second dense layer
dense2.forward(activation1.output)
# Make a forward pass through the second activation function
activation2.forward(dense2.output)
# Save the output
output = activation2.output
print(output[:5])








