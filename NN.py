import numpy as np
import nnfs
from nnfs.datasets import spiral_data  
nnfs.init()
import matplotlib.pyplot as plt

#Dense Layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs , n_neurons)             # weights matrix defined as inputs x neurons initialized with random numbers from normal distribution mu = 0 and sigma = 1
        self.biases = np.zeros((1, n_neurons))                                # bias is a column vector

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):  # dvalues is dL/dz where z is input*weights + bias
        # Gradient of Loss respect to weights and biases
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        # Gradient of Loss respect to inputs x
        self.dinputs = np.dot(dvalues, self.weights.T)

class activation_reLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):  # dvalues represent dL/da where a is ReLU(z)   
        #inputs are z1, z2, z3
        # we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()  #dinputs represent dL/dz
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #unnormalized probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#combined softmax activvation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = activation_softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        self.output = self.activation.output 
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):    #dvalues are the predicted values, our goal is to find dL/dz where z is the input of the aactivation function
        # we go from one-hot envoded to number representation
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis= 1)        
        # copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -=1     #dinputs is dL/dz 
        # Normalize gradient 
        self.dinputs = self.dinputs / samples

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss             

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        #number of samples in a batch
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) #clip data to prevent division by 0
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):   # dvalues here are the predicted outputs
        #number of samples
        samples = len(dvalues)
        #number of labels in every sample
        #we'll use the first sample to count them
        labels = len(dvalues[0])
        
        #one-hot encoding if the labels are sparse
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # calculate gradient
        self.dinputs = -y_true/dvalues    # dinputs represent dL/dy_true
        # Normalize gradient
        self.dinputs = self.dinputs/samples

#SDG optimizer 
class Optimizer_SDG:
    def __init__(self, learning_rate = 1
                 ):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases  

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,1000)
# Create ReLU activation
activation1 = activation_reLU()
# Create second Dense layer with 3 input features and 3 output values
dense2 = Layer_Dense(1000,3)
# Create combined Softmax activation and categorical cross-entropy loss
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_SDG()

for epoch in range(10001):
    # Make a forward pass of our training data through the first layer
    dense1.forward(X)
    # Make a forward pass through the first activation function
    activation1.forward(dense1.output)

    # Make a forward pass through the second dense layer
    dense2.forward(activation1.output)
    # Make a forward pass through the combined softmax activation and loss
    loss = loss_activation.forward(dense2.output, y)

    # Check accuracy  

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    # True evaluates to 1; False  evaluates to 0
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(
            f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}'
        )
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
