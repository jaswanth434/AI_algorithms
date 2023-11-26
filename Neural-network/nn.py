import numpy as np


def read_from_terminal(prompt, data_type):
    return data_type(input(prompt))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
    def __init__(self, layers, learning_rate, perceptron_bias):
        self.learning_rate = learning_rate
        self.bias = perceptron_bias
        self.weights = []
        for x, y in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.randn(y, x + 1))  # +1 for bias
        self.previous_loss = 0

    def feedforward(self, X):
        activations = [X]
        for i, w in enumerate(self.weights):
            net_input = np.dot(activations[-1], w[:, :-1].T) + w[:, -1] * self.bias
            activation = relu(net_input) if i < len(self.weights) - 1 else sigmoid(net_input)
            activations.append(activation)
        return activations

    def backprop(self, X, y, activations):
        deltas = [y - activations[-1]]
        for l in range(len(activations) - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[l][:, :-1]) * relu_derivative(activations[l])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            layer = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i][:, :-1] += self.learning_rate * np.dot(delta.T, layer)
            self.weights[i][:, -1] += self.learning_rate * delta.sum(axis=0) * self.bias

    def train(self, X, y, epochs, adjustment_threshold):
        for epoch in range(epochs):
            activations = self.feedforward(X)
            self.backprop(X, y, activations)

            # if epoch % 100 == 0:
            loss = mse_loss(y, activations[-1])
            print(f'Epoch {epoch}, Loss: {loss} , Loss difference : {abs(self.previous_loss - loss):.10f}')
            if self.previous_loss is not None:
                loss_difference = abs(self.previous_loss - loss)
                if loss_difference < adjustment_threshold:
                    print(f"Converged. Loss difference ({loss_difference}) is less than the threshold ({adjustment_threshold}).")
                    break
            self.previous_loss = loss

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
nn = NeuralNetwork(layers, learning_rate, perceptron_bias)

# Generate random training data
NUM_DATA_POINTS = 100
input_sequences = np.random.rand(NUM_DATA_POINTS, input_size)
output_sequences = np.random.rand(NUM_DATA_POINTS, output_size)
training_data = list(zip(input_sequences, output_sequences))

# Prepare data for training
X, y = zip(*training_data)
X, y = np.array(X), np.array(y)

# Train the network
nn.train(X, y, max_cycles, adjustment_threshold)

# Test the network with a new data point
test_data = np.random.rand(1, input_size)
output = nn.feedforward(test_data)[-1]
print("Test Output:", output)