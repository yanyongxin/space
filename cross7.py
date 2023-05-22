'''
Created on May 19, 2023

@author: yanyo
'''
'''
Created on May 18, 2023

@author: yanyo
'''
from tensorflow.python.ops.gen_experimental_dataset_ops import ComputeBatchSize
'''
Created on May 18, 2023

@author: yanyo
'''
from datetime import datetime
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)




# Define the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the softmax activation function for the output layer
def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, dropoutRate, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropoutRate
        self.output_size = output_size
        self.mask = None
        
        # Initialize the weights with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize the biases with zeros
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
        
    def dropout(self, inputs, training=True):
        if training:
            # Generate a binary mask with the same shape as inputs
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
            # Scale the outputs by the inverted dropout rate
            outputs = inputs * self.mask / (1 - self.dropout_rate)
        else:
            # During inference, multiply inputs by the keep probability
            outputs = inputs * (1 - self.dropout_rate)
        
        return outputs

    def forward(self, X, training=True):
        # Forward propagation
        self.hidden_layer = relu(np.dot(X, self.W1) + self.b1)
        self.afterdropout = self.dropout(self.hidden_layer, training)
        self.output_layer = softmax(np.dot(self.afterdropout, self.W2) + self.b2)
        
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # Compute the gradients
        output_error = self.output_layer - y
        
        dW2 = self.hidden_layer.T.dot(output_error)
        db2 = np.sum(output_error, axis=0, keepdims=True)
        
        hidden = self.hidden_layer*self.mask / (1 - self.dropout_rate)

        hidden_error = output_error.dot(self.W2.T) * relu_derivative(hidden)
        
        dW1 = X.T.dot(hidden_error)
        db1 = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Update the weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def calculate_cost(self, X, y):
        # Calculate the cross-entropy cost
        epsilon = 1e-15
        num_examples = len(X)
        corect_logprobs = 0.0
        for i in range(num_examples):
            yy = y[i]
            out = self.output_layer[i]
            outsize = len(out)
            for j in range(outsize):
                out_out = out[j]+epsilon
                logout = np.log(out_out)
                corect_logprobs -= yy[j]*logout
#        corect_logprobs = -np.log(out)
#        cost = np.sum(corect_logprobs) / num_examples
        cost = corect_logprobs/num_examples
        return cost

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            # Generate a binary mask with the same shape as inputs
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
            # Scale the outputs by the inverted dropout rate
            outputs = inputs * self.mask / (1 - self.dropout_rate)
        else:
            # During inference, multiply inputs by the keep probability
            outputs = inputs * (1 - self.dropout_rate)
        
        return outputs

    def backward(self, grad):
        # Backward pass propagating the gradient through the dropout layer
        grad *= self.mask / (1 - self.dropout_rate)
        return grad

start_time = datetime.now()
print("start_time:", start_time)

mnist = tf.keras.datasets.mnist

(xx_train, yy_train), (xx_test, yy_test) = mnist.load_data()
xx_train, xx_test = xx_train / 255.0, xx_test / 255.0

print("xx_train: ", xx_train.shape)
print("yy_train: ", yy_train.shape)
print("xx_test: ", xx_test.shape)
print("yy_test: ", yy_test.shape)

# Convert the labels to one-hot encoding
num_classes = 10
yyy_train = np.eye(num_classes)[yy_train]
yyy_test = np.eye(num_classes)[yy_test]

nTrain = len(yyy_train)
nTest = len(yyy_test)

xxx_train = xx_train.reshape((nTrain, 784))
xxx_test = xx_test.reshape((nTest, 784))

x_test = xxx_test
y_test = yyy_test

nTrain = len(yyy_train)
nTest = len(y_test)

print(f"nTrain: {nTrain}, nTest: {nTest}")
# Create a neural network with 2 input neurons, 2 hidden neurons, and 2 output neurons
nn = NeuralNetwork(784, 128, 0.2, num_classes)

# Train the neural network using backpropagation
epochs = 200
learning_rate = 0.001
tolerance = 1e-6
prev_cost = np.inf

batchSize = 100
nBatches = nTrain//batchSize
for i in range(epochs):
    startPos = 0
    endPos = startPos + batchSize
    cost = 0
    for j in range(nBatches):
        x_train = xxx_train[startPos:endPos]
        y_train = yyy_train[startPos:endPos]

        nn.forward(x_train)
        nn.backward(x_train, y_train, learning_rate)
    
    # Calculate the cost and check for convergence
        cost += nn.calculate_cost(x_train, y_train)
        
    cost /= nBatches
    print(f"Epoch: {i}, cost: {cost}")
    nn.forward(x_test, training=False)    
    cost = nn.calculate_cost(x_test, y_test)
    print("Test cost: ", cost)

#    if abs(cost - prev_cost) < tolerance:
#        print(f"Converged after {i+1} epochs.")
#        break
    
    prev_cost = cost
    
# Test the trained neural network
nn.forward(x_test)    
cost = nn.calculate_cost(x_test, y_test)
print("Test cost: ", cost)

end_time = datetime.now()
print("end_time:", end_time)
time_difference = (end_time - start_time).total_seconds()
print("Duration in seconds:", time_difference)

