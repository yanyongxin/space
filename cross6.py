'''
Created on May 19, 2023

@author: yanyo
'''
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
        
        # Initialize the Adam optimization parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.mW1 = np.zeros_like(self.W1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.mb1 = np.zeros_like(self.b1)
        self.mb2 = np.zeros_like(self.b2)
            
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
          
        # Update the Adam optimization parameters
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        
        # Bias-corrected parameter updates
        mW1_hat = self.mW1 / (1 - self.beta1)
        vW1_hat = self.vW1 / (1 - self.beta2)
        mW2_hat = self.mW2 / (1 - self.beta1)
        vW2_hat = self.vW2 / (1 - self.beta2)
        mb1_hat = self.mb1 / (1 - self.beta1)
        mb2_hat = self.mb2 / (1 - self.beta1)
        
        # Update the weights and biases using Adam optimization
        self.W1 -= learning_rate * mW1_hat / (np.sqrt(vW1_hat) + self.epsilon)
        self.W2 -= learning_rate * mW2_hat / (np.sqrt(vW2_hat) + self.epsilon)
        self.b1 -= learning_rate * mb1_hat
        self.b2 -= learning_rate * mb2_hat

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
nn = NeuralNetwork(2, 10, num_classes)

# Train the neural network using backpropagation
epochs = 10000
learning_rate = 0.021
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
