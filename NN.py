import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
import matplotlib.pyplot as plt
from nn_classes import (
    Layer,
    activation_reLU,
    activation_softmax,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Loss,
    Loss_CategoricalCrossentropy,
    Optimizer_SGD,
    Learning_Decay_Rate,
    Momentum_Optimizer,
    Adagrad_Optimizer,
    Adam_Optimizer,
    Layer_Dropout
)

Neurons = 64
epochs = 10001

# Create dataset
#  
X, y = spiral_data(samples=300, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer(2,Neurons, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
# Create ReLU activation
activation1 = activation_reLU()
# Create second Dense layer with 3 input features and 3 output values
dense2 = Layer(Neurons,3)
# Create combined Softmax activation and categorical cross-entropy loss
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#optimizer = Adagrad_Optimizer(decay=1e-4)
#optimizer = Momentum_Optimizer(decay=1e-3, momentum=0.9)
optimizer = Adam_Optimizer(decay=5e-7)

for epoch in range(epochs):
    # Make a forward pass of our training data through the first layer
    dense1.forward(X)
    # Make a forward pass through the first activation function
    activation1.forward(dense1.output)

    dropout1 = Layer_Dropout(0.1)

    # Make a forward pass through the second dense layer
    dense2.forward(activation1.output)
    # Make a forward pass through the combined softmax activation and loss
    data_loss = loss_activation.forward(dense2.output, y)
    # Calculate regularization loss
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2) 
    loss = data_loss + regularization_loss

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


    optimizer.pre_update_params()    
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model
# Create dataset

X_val, y_val = spiral_data(samples=100, classes=3)
# Make a forward pass of our validation data through the first layer
dense1.forward(X_val)
# Make a forward pass through the first activation function
activation1.forward(dense1.output)
# Make a forward pass through the second dense layer
dense2.forward(activation1.output)
# Make a forward pass through the combined softmax activation and loss to calculate the loss from output of activation2 
data_loss = loss_activation.forward(dense2.output, y_val)
# Calculate regularization loss
regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)

# Total loss is data loss and regularization loss
loss = data_loss + regularization_loss

# Check accuracy
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_val.shape) == 2:
    y_val = np.argmax(y_val, axis=1)
accuracy = np.mean(predictions == y_val)
print(
    f'validation, ' +
    f'acc: {accuracy:.3f}, ' +
    f'loss: {loss:.3f}'
)