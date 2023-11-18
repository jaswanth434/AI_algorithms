import numpy as np
import math
import json

# Configuration
EPSILON = 0.00001
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0
TAU_FIRE = 1.0  # Define a threshold for the activation function

# Helper Functions
def read_from_terminal(prompt, value_type):
    while True:
        try:
            return value_type(input(prompt))
        except ValueError:
            print("Invalid input, please enter a valid number.")

def modify_weights(edges, layer, delta, rate):
    new_edges = []
    for (source, target, weight) in edges:
        if layer == source:
            updated_weight = weight * (1 + rate * delta)
            new_edges.append((source, target, updated_weight))
        else:
            new_edges.append((source, target, weight))
    return new_edges

def insert_updated_edges(updated_edges, layer, layer_matrices, num_layers, num_nodes):
    updated_matrices = []
    updated_edges.sort(key=lambda x: (x[0], x[1]))

    for i in range(layer):
        updated_matrices.append(layer_matrices[i])

    new_matrix = np.zeros((num_nodes, num_nodes))
    for (source, target, weight) in updated_edges:
        new_matrix[source][target] = weight
    updated_matrices.append(new_matrix)

    for i in range(layer + 1, num_layers):
        updated_matrices.append(layer_matrices[i])

    return updated_matrices

def get_relevant_edges(target_layer, current_layer, edges, matrices, num_nodes):
    if current_layer == target_layer:
        return edges
    else:
        source_nodes = {edge[0] for edge in edges}
        destination_nodes = {edge[1] for edge in edges}
        layer_matrix = matrices[current_layer]

        new_edges = set()
        for source in range(num_nodes):
            if source in source_nodes:
                for target in range(num_nodes):
                    if target in destination_nodes:
                        weight = layer_matrix[source][target]
                        if abs(weight) > EPSILON:
                            new_edges.add((source, target, weight))

        return get_relevant_edges(target_layer, current_layer - 1, new_edges, matrices, num_nodes)

def threshold_activation(inputs, threshold):
    outputs = []
    for value in inputs:
        if value > threshold:
            outputs.append(1 / (1 + math.exp(-value)))
        else:
            outputs.append(0)
    return outputs

def process_layers(start_layer, end_layer, matrices, previous_output, bias_hidden, bias_output):
    print(f"Start Layer: {start_layer}, End Layer: {end_layer}")
    print(f"Initial Previous Output Shape: {previous_output.shape}")

    for layer in range(start_layer, end_layer):
        print(f"Processing Layer: {layer}")
        print(f"Matrix Shape: {matrices[layer].shape}, Previous Output Shape: {previous_output.shape}")
        input_values = np.dot(matrices[layer], previous_output) + bias_hidden
        previous_output = threshold_activation(input_values, TAU_FIRE)
        print(f"New Previous Output Shape: {previous_output.shape}")

    print(f"Final Layer Matrix Shape: {matrices[end_layer].shape}, Previous Output Shape: {previous_output.shape}")
    final_input = np.dot(matrices[end_layer], previous_output) + bias_output
    final_output = threshold_activation(final_input, TAU_FIRE)
    return final_output


def adjust_network_weights(actual_output, computed_output, matrices, max_iterations, threshold_adj, learning_rate, num_hidden_layers, output_size, num_nodes):
    current_layer = num_hidden_layers
    minimum_error = float('inf')
    iteration = 1
    adjusted = False

    while current_layer > 0 and iteration <= max_iterations:
        delta = [actual_output[i] - computed_output[i] for i in range(len(actual_output))]
        significant_differences = [(i, d) for i, d in enumerate(delta) if abs(d) > EPSILON]

        relevant_edges = get_relevant_edges(current_layer, num_hidden_layers, significant_differences, matrices, num_nodes)
        updated_edges = modify_weights(relevant_edges, current_layer, delta, learning_rate)
        new_matrices = insert_updated_edges(updated_edges, current_layer, matrices, num_hidden_layers, num_nodes)
        computed_output = process_layers(1, num_hidden_layers, new_matrices, bias_vector_hidden, bias_vector_output)

        error = sum([(actual_output[j] - computed_output[j])**2 for j in range(output_size)])
        if error < minimum_error:
            matrices = new_matrices
            minimum_error = error
            if error < threshold_adj:
                adjusted = True
                break
        iteration += 1

    return matrices if adjusted else current_layer - 1

def create_random_matrix(rows, columns):
    return np.random.uniform(LOWER_BOUND, UPPER_BOUND, (rows, columns))



def FeedForward_Neural_Network(input_size, output_size, training_data):
    num_perceptrons_hidden = read_from_terminal("Number of perceptrons in each hidden layer:", int)
    num_hidden_layers = read_from_terminal("Number of hidden layers:", int)
    perceptron_bias = read_from_terminal("Perceptron bias:", float)
    max_cycles = read_from_terminal("Cycles to be repeated:", int)
    adjustment_threshold = read_from_terminal("Threshold for neural network adjustment:", float)
    learning_rate = read_from_terminal("Learning rate:", float)

    # Configuration - modify as needed
  

    input_sequences, output_sequences = zip(*training_data)
    trained_matrices = [create_random_matrix(input_size, num_perceptrons_hidden)]
    trained_matrices.extend([create_random_matrix(num_perceptrons_hidden, num_perceptrons_hidden) for _ in range(1, num_hidden_layers)])
    trained_matrices.append(create_random_matrix(num_perceptrons_hidden, output_size))

    bias_vector_input = np.zeros(input_size)
    bias_vector_hidden = np.full(num_perceptrons_hidden, perceptron_bias)
    bias_vector_output = np.zeros(output_size)

    for iteration in range(max_cycles):
        for input_vector, output_vector in zip(input_sequences, output_sequences):
            initial_input = np.add(input_vector, bias_vector_input)
            initial_output = threshold_activation(initial_input, TAU_FIRE)
            computed_output = process_layers(0, num_hidden_layers, trained_matrices, initial_output, bias_vector_hidden, bias_vector_output)

            error = sum((computed - actual) ** 2 for computed, actual in zip(computed_output, output_vector))
            if error > adjustment_threshold:
                trained_matrices = adjust_network_weights(output_vector, computed_output, trained_matrices, max_cycles, adjustment_threshold, learning_rate, num_hidden_layers, output_size, num_perceptrons_hidden)
            else:
                print('Adjustment complete')
                return trained_matrices

    print('Cycles over')
    return trained_matrices

def read_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # input_size = data["input_size"]
    # output_size = data["output_size"]
    training_data = [(pair["input"], pair["output"]) for pair in data]

    return training_data

def generate_test_data(input_size, output_size, num_samples=100):
    test_data = []
    for _ in range(num_samples):
        # Generate random inputs and outputs
        input_sample = np.random.rand(input_size).tolist()  # Convert to Python list
        output_sample = np.random.randint(0, 2, output_size).tolist()  # Convert to Python list

        # Convert elements to native Python types
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
        # Generate and save test data
        test_data = generate_test_data(input_size, output_size, 200)
        save_to_json(test_data, 'test_data.json')

    training_data = read_json_data('test_data.json')
    trained_matrices = FeedForward_Neural_Network(input_size, output_size, training_data)
    print("Training complete. Trained matrices:", trained_matrices)

if __name__ == "__main__":
    main()
