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

# Define the softmax activation function for the output layer
def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

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
        self.output_layer = softmax(np.dot(self.hidden_layer, self.W2) + self.b2)
        
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # Compute the gradients
        output_error = self.output_layer - y
        
        dW2 = self.hidden_layer.T.dot(output_error)
        db2 = np.sum(output_error, axis=0, keepdims=True)
        
        hidden_error = output_error.dot(self.W2.T) * relu_derivative(self.hidden_layer)
        
        dW1 = X.T.dot(hidden_error)
        db1 = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Update the weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def calculate_cost(self, X, y):
        # Calculate the cross-entropy cost
        num_examples = len(X)
        corect_logprobs = 0.0
        for i in range(num_examples):
            yy = y[i]
            out = self.output_layer[i]
            corect_logprobs += -yy[0]*np.log(out[0]) - yy[1]*np.log(out[1])
#        corect_logprobs = -np.log(out)
#        cost = np.sum(corect_logprobs) / num_examples
        cost = corect_logprobs/num_examples
        return cost

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # Updated to single-label format

# Convert the labels to one-hot encoding
num_classes = len(np.unique(y))
y_onehot = np.eye(num_classes)[y]

# Create a neural network with 2 input neurons, 2 hidden neurons, and 2 output neurons
nn = NeuralNetwork(2, 2, num_classes)

# Train the neural network using backpropagation
epochs = 10000
learning_rate = 0.1
tolerance = 1e-4
prev_cost = np.inf

for i in range(epochs):
    nn.forward(X)
    nn.backward(X, y_onehot, learning_rate)
    
    # Calculate the cost and check for convergence
    cost = nn.calculate_cost(X, y_onehot)
    print(f"Cost: {cost}")
#    if abs(cost - prev_cost) < tolerance:
#        print(f"Converged after {i+1} epochs.")
#        break
    
#    prev_cost = cost

# Test the trained neural network
nn.forward(X)
predicted_labels = np.argmax(nn.output_layer, axis=1)
print("Predicted Output:\n", predicted_labels)
