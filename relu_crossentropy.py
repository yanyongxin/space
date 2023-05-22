'''
Created on May 18, 2023

@author: yanyo
'''
import numpy as np

# Define the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize the weights with random values
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize the biases with zeros
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        # Forward propagation
        self.hidden_layer = relu(np.dot(X, self.W1) + self.b1)
        self.output_layer = relu(np.dot(self.hidden_layer, self.W2) + self.b2)
        
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # Compute the gradients
        output_error = y - self.output_layer
        output_delta = output_error * relu_derivative(self.output_layer)
        
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * relu_derivative(self.hidden_layer)
        
        # Update the weights and biases
        self.W2 += self.hidden_layer.T.dot(output_delta) * learning_rate
        self.W1 += X.T.dot(hidden_delta) * learning_rate
        
        self.b2 += np.sum(output_delta, axis=0) * learning_rate
        self.b1 += np.sum(hidden_delta, axis=0) * learning_rate

    def calculate_cost(self, X, y):
        # Calculate the cross-entropy cost
        num_examples = len(X)
        h1 = np.arange(num_examples)
        y1 = y.flatten()
        out = self.output_layer[h1, y1]
        log_probs = -np.log(out)
        cost = np.sum(log_probs) / num_examples
        return cost

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = NeuralNetwork(2, 2, 1)

# Train the neural network using backpropagation
epochs = 10000
learning_rate = 0.1
tolerance = 1e-4
prev_cost = np.inf

for i in range(epochs):
    nn.forward(X)
    nn.backward(X, y, learning_rate)
    
    # Calculate the cost and check for convergence
    cost = nn.calculate_cost(X, y)
    if abs(cost - prev_cost) < tolerance:
        print(f"Converged after {i+1} epochs.")
        break
        
    prev_cost = cost

# Test the trained neural network
nn.forward(X)
print("Predicted Output:\n", nn.output_layer)
