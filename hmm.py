import numpy as np
import itertools

def read_hmm_parameters(file_path):
    """
    Reads HMM parameters and the emission sequence from a file.
    """
    with open(file_path, 'r') as file:
        num_states = int(file.readline().strip())
        num_emissions = int(file.readline().strip())
        
        transition_matrix = np.array([list(map(float, file.readline().split())) for _ in range(num_states)])
        emission_matrix = np.array([list(map(float, file.readline().split())) for _ in range(num_states)])
        initial_probabilities = np.array(list(map(float, file.readline().split())))
        emission_sequence = list(map(int, file.readline().split()))

    return num_states, num_emissions, transition_matrix, emission_matrix, initial_probabilities, emission_sequence


def is_valid_emission(sequence, num_emissions):
    """
    Checks if the emission sequence is valid.
    """
    return all(0 <= e < num_emissions for e in sequence)

def is_valid_transition(state_sequence, transition_matrix, initial_probabilities):
    """
    Checks if the state sequence is valid based on the transition matrix and initial probabilities.
    """
    if initial_probabilities[state_sequence[0]] == 0:
        return False

    for i in range(1, len(state_sequence)):
        if transition_matrix[state_sequence[i-1], state_sequence[i]] == 0:
            return False

    return True

def calculate_probability(state_sequence, emission_sequence, transition_matrix, emission_matrix, initial_probabilities):
    """
    Calculates the probability of a state sequence emitting the given emission sequence.
    """
    probability = initial_probabilities[state_sequence[0]] * emission_matrix[state_sequence[0], emission_sequence[0]]

    for i in range(1, len(state_sequence)):
        trans_prob = transition_matrix[state_sequence[i-1], state_sequence[i]]
        emit_prob = emission_matrix[state_sequence[i], emission_sequence[i]]
        probability *= trans_prob * emit_prob

    return probability

def find_most_probable_path(emission_sequence, num_states, transition_matrix, emission_matrix, initial_probabilities):
    """
    Finds the most probable path for the given emission sequence.
    """
    max_probability = 0
    most_probable_sequence = None

    # Generate all possible state sequences
    for state_sequence in itertools.product(range(num_states), repeat=len(emission_sequence)):
        if is_valid_transition(state_sequence, transition_matrix, initial_probabilities):
            probability = calculate_probability(state_sequence, emission_sequence, transition_matrix, emission_matrix, initial_probabilities)
            
            if probability > max_probability:
                max_probability = probability
                most_probable_sequence = state_sequence

    return most_probable_sequence

def main():
    file_path = input("Enter the file path for HMM parameters: ")
    num_states, num_emissions, transition_matrix, emission_matrix, initial_probabilities, emission_sequence = read_hmm_parameters(file_path)

    print("Read Transition Matrix:\n", transition_matrix)
    print("Read Emission Matrix:\n", emission_matrix)
    print("Read Initial State Probabilities:\n", initial_probabilities)
    print("Read Emission Sequence:", emission_sequence)

    if not is_valid_emission(emission_sequence, num_emissions):
        print("Invalid emission sequence.")
        return

    most_probable_path = find_most_probable_path(emission_sequence, num_states, transition_matrix, emission_matrix, initial_probabilities)
    print("Most Probable Path:", most_probable_path)

if __name__ == "__main__":
    main()
