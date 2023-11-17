import numpy as np

def read_hmm_parameters(file_path):
    """
    Reads HMM parameters from a file.
    Expected file format:
    - First line: Number of states
    - Second line: Number of emissions
    - Next 'number of states' lines: Transition matrix rows
    - Next 'number of states' lines: Emission matrix rows
    - Next line: Initial state probabilities
    """
    with open(file_path, 'r') as file:
        num_states = int(file.readline().strip())
        num_emissions = int(file.readline().strip())
        
        transition_matrix = np.array([list(map(float, file.readline().split())) for _ in range(num_states)])
        emission_matrix = np.array([list(map(float, file.readline().split())) for _ in range(num_states)])
        initial_probabilities = np.array(list(map(float, file.readline().split())))

    return num_states, num_emissions, transition_matrix, emission_matrix, initial_probabilities


def input_matrix(rows, cols, message):
    """
    Function to input a matrix from the user.
    """
    print(f"Enter the {message} matrix ({rows}x{cols}):")
    return np.array([list(map(float, input(f"Row {i+1}: ").split())) for i in range(rows)])

def input_vector(length, message):
    """
    Function to input a vector from the user.
    """
    print(f"Enter the {message} vector (length {length}):")
    return np.array(list(map(float, input().split())))


def is_valid_emission(sequence, num_emissions):
    """
    Checks if the emission sequence is valid (each emission should be within the range of defined emissions).
    """
    return all(0 <= e < num_emissions for e in sequence)

def calculate_probability(path, sequence, transition_matrix, emission_matrix, initial_probabilities):
    """
    Calculates the probability of a given path producing the given emission sequence.
    """
    prob = initial_probabilities[path[0]] * emission_matrix[path[0], sequence[0]]
    for i in range(1, len(path)):
        prob *= transition_matrix[path[i-1], path[i]] * emission_matrix[path[i], sequence[i]]
    return prob

def find_most_probable_path(emission_sequence, num_states, transition_matrix, emission_matrix, initial_probabilities):
    """
    Finds the most probable path for the given emission sequence using the Viterbi algorithm.
    """
    len_sequence = len(emission_sequence)
    viterbi_matrix = np.zeros((num_states, len_sequence))
    path_matrix = np.zeros((num_states, len_sequence), dtype=int)

    # Initialize first column of viterbi matrix
    for s in range(num_states):
        viterbi_matrix[s, 0] = initial_probabilities[s] * emission_matrix[s, emission_sequence[0]]
    
    # Fill the viterbi matrix and path matrix
    for t in range(1, len_sequence):
        for s in range(num_states):
            prob, state = max((viterbi_matrix[s_prev, t - 1] * transition_matrix[s_prev, s] * emission_matrix[s, emission_sequence[t]], s_prev) for s_prev in range(num_states))
            viterbi_matrix[s, t] = prob
            path_matrix[s, t] = state
    
    # Backtrack to find the most probable path
    last_state = np.argmax(viterbi_matrix[:, len_sequence - 1])
    path = [last_state]
    for t in range(len_sequence - 1, 0, -1):
        last_state = path_matrix[last_state, t]
        path.insert(0, last_state)
    
    return path

def generate_emission_sequence(num_states, num_emissions, transition_matrix, emission_matrix, initial_probabilities, sequence_length):
    """
    Generates an emission sequence of the given length based on HMM parameters.
    """
    # Choose the initial state based on initial probabilities
    current_state = np.random.choice(num_states, p=initial_probabilities)
    emission_sequence = []

    for _ in range(sequence_length):
        # Generate emission based on the emission matrix of the current state
        emission = np.random.choice(num_emissions, p=emission_matrix[current_state])
        emission_sequence.append(emission)

        # Transition to the next state based on the transition matrix
        current_state = np.random.choice(num_states, p=transition_matrix[current_state])

    return emission_sequence

def generate_random_probability_vector(length):
    """
    Generates a random probability vector of a given length.
    """
    vector = np.random.rand(length)
    return vector / vector.sum()

def generate_random_probability_matrix(rows, cols):
    """
    Generates a random probability matrix of the given size.
    """
    matrix = np.array([generate_random_probability_vector(cols) for _ in range(rows)])
    return matrix

def main():
    num_states = int(input("Enter the number of states: "))
    num_emissions = int(input("Enter the number of emissions: "))

    transition_matrix = generate_random_probability_matrix(num_states, num_states)
    emission_matrix = generate_random_probability_matrix(num_states, num_emissions)
    initial_probabilities = generate_random_probability_vector(num_states)

    print("Generated Transition Matrix:\n", transition_matrix)
    print("Generated Emission Matrix:\n", emission_matrix)
    print("Generated Initial State Probabilities:\n", initial_probabilities)

    sequence_length = int(input("Enter the length of the emission sequence to be generated: "))

    # Generate the emission sequence
    emission_sequence = generate_emission_sequence(num_states, num_emissions, transition_matrix, emission_matrix, initial_probabilities, sequence_length)
    print("Generated Emission Sequence:", emission_sequence)

    # Calculate the most probable path
    most_probable_path = find_most_probable_path(emission_sequence, num_states, transition_matrix, emission_matrix, initial_probabilities)
    print("Most Probable Path:", most_probable_path)

if __name__ == "__main__":
    main()
