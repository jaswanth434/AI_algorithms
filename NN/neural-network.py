import numpy as np
import math
import json

EPSILON = 0.00001
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0
TAU_FIRE = 1.0 

def read_from_terminal(prompt, value_type):
    while True:
        try:
            return value_type(input(prompt))
        except ValueError:
            print("Invalid input, please enter a valid number.")


def modify_weights(edges, layer, delta, rate):
    print("Entering modify_weights function")
    new_edges = []
    print(edges)
    for (source, target, weight) in edges:
        print(f"Edge: {(source, target, weight)}")
        if layer == source:
            updated_weight = weight * (1 + rate * delta)
            new_edges.append((source, target, updated_weight))
        else:
            new_edges.append((source, target, weight))
    print("Exiting modify_weights function")
    return new_edges


def threshold_activation(inputs, threshold):
    outputs = []
    for value in inputs:
        if value > threshold:
            outputs.append(1 / (1 + math.exp(-value)))
        else:
            outputs.append(0)
    return outputs


def create_random_matrix(rows, columns):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (rows, columns))


def FeedForward_Neural_Network(input_size, output_size, training_data):
    num_perceptrons_hidden = read_from_terminal("Number of perceptrons in each hidden layer:", int)
    num_hidden_layers = read_from_terminal("Number of hidden layers:", int)
    perceptron_bias = read_from_terminal("Perceptron bias:", float)
    max_cycles = read_from_terminal("Cycles to be repeated:", int)
    adjustment_threshold = read_from_terminal("Threshold for neural network adjustment:", float)
    learning_rate = read_from_terminal("Learning rate:", float)

    input_sequences, output_sequences = zip(*training_data)
    trained_matrices = [create_random_matrix(input_size, num_perceptrons_hidden)]

    print(f"Trained Matrices: {trained_matrices}")
   
    return trained_matrices

def read_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # input_size = data["input_size"]
    # output_size = data["output_size"]
    training_data = [(pair["input"], pair["output"]) for pair in data]

    return training_data

def generate_test_data(input_size, output_size, num_samples=200):
    test_data = []
    for _ in range(num_samples):
        input_sample = np.random.rand(input_size).tolist() 
        output_sample = np.random.randint(0, 2, output_size).tolist() 

        
        input_sample = [float(i) for i in input_sample]
        output_sample = [int(o) for o in output_sample]

        test_data.append({"input": input_sample, "output": output_sample})
    
    return test_data


def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def main():
      
    input_size = read_from_terminal("Input size: ", int) 
    output_size = read_from_terminal("output size: ", int) 

    gen_test_data = read_from_terminal("Generate test data ? (y|n): ", str)

    if gen_test_data == "y":
        test_data = generate_test_data(input_size, output_size, 200)
        save_to_json(test_data, 'test_data.json')

    training_data = read_json_data('test_data.json')
    trained_matrices = FeedForward_Neural_Network(input_size, output_size, training_data)
    print("Training complete. Trained matrices:", trained_matrices)

if __name__ == "__main__":
    main()
