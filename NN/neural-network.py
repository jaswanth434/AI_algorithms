import numpy as np

def read_from_terminal(prompt, data_type):
    return data_type(input(prompt))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def feedforward(self, X):
        activations = [X]
        for i, w in enumerate(self.weights):
            net_input = np.dot(activations[-1], w.T)
            activation = relu(net_input) if i < len(self.weights) - 1 else sigmoid(net_input)
            activations.append(activation)
        return activations

    def backprop(self, X, y, activations):
        deltas = [y - activations[-1]]
        for l in range(len(activations) - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[l]) * relu_derivative(activations[l])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            layer = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += self.learning_rate * np.dot(delta.T, layer)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            activations = self.feedforward(X)
            self.backprop(X, y, activations)

            if epoch % 100 == 0:
                loss = mse_loss(y, activations[-1])
                print(f'Epoch {epoch}, Loss: {loss}')

# Reading configuration from terminal
num_perceptrons_hidden = read_from_terminal("Number of perceptrons in each hidden layer: ", int)
num_hidden_layers = read_from_terminal("Number of hidden layers: ", int)
perceptron_bias = read_from_terminal("Perceptron bias: ", float)
max_cycles = read_from_terminal("Cycles to be repeated: ", int)
adjustment_threshold = read_from_terminal("Threshold for neural network adjustment: ", float)
learning_rate = read_from_terminal("Learning rate: ", float)
input_size = read_from_terminal("Input size: ", int)
output_size = read_from_terminal("Output size: ", int)

# Configure neural network layers
layers = [input_size] + [num_perceptrons_hidden] * num_hidden_layers + [output_size]

# Initialize neural network
nn = NeuralNetwork(layers, learning_rate)

# Generate random training data
NUM_DATA_POINTS = 100
input_sequences = np.random.rand(NUM_DATA_POINTS, input_size)
output_sequences = np.random.rand(NUM_DATA_POINTS, output_size)
training_data = list(zip(input_sequences, output_sequences))

# Train the network
X, y = zip(*training_data)
nn.train(np.array(X), np.array(y), max_cycles)

# Test the network with a new data point
test_data = np.random.rand(1, input_size)
output = nn.feedforward(test_data)[-1]
print("Test Output:", output)
