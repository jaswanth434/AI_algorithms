import math
import itertools
import json

def load_hmm_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def is_valid_emission(emission_sequence, emission_symbols):
    return all(symbol in emission_symbols for symbol in emission_sequence)

def is_valid_transition(sequence, transition_matrix, state_symbols, initial_probabilities):
    n = len(sequence)
    m = 0
    is_possible = True
    
    if abs(initial_probabilities[state_symbols.index(sequence[0])]) < 1e-10:
        is_possible = False
    
    while m < n - 1 and is_possible:
        i = state_symbols.index(sequence[m])
        j = state_symbols.index(sequence[m + 1])
        
        if abs(transition_matrix[i][j]) < 1e-10:
            is_possible = False
        
        m += 1
    
    return is_possible

def find_emission_states(emission, emission_matrix, emission_symbols, num_states):
    emission_index = emission_symbols.index(emission)
    emitting_states = set()
    
    for i in range(num_states):
        if abs(emission_matrix[i][emission_index]) >= 1e-10:
            emitting_states.add(i)
    
    return emitting_states

def calculate_sequence_probability(state_sequence, emission_sequence, state_symbols, emission_symbols, transition_matrix, emission_matrix, initial_probabilities):
    sequence_length = len(state_sequence)
    probability = initial_probabilities[state_symbols.index(state_sequence[0])]

    for i in range(sequence_length):
        row_transition = state_symbols.index(state_sequence[i - 1]) if i > 0 else -1
        col_transition = state_symbols.index(state_sequence[i])
        col_emission = emission_symbols.index(emission_sequence[i])

        prob_transition = 1.0 if row_transition == -1 else transition_matrix[row_transition][col_transition]
        prob_emission = emission_matrix[col_transition][col_emission]

        probability *= prob_transition * prob_emission

    return probability

def hmm_path_finder():
    filepath = input("Enter the file path for HMM data: ")
    data = load_hmm_data(filepath)
    
    state_symbols = data['states']
    emission_symbols = data['emissions']
    transition_matrix = data['transition_matrix']
    emission_matrix = data['emission_matrix']
    initial_probabilities = data['initial_probabilities']

    while True:
        emission_sequence = input("Enter the next emission sequence (type 'exit' to quit): ")
        
        if emission_sequence == "exit":
            break
        
        if is_valid_emission(emission_sequence, emission_symbols):
            possible_transitions = []
            emission_sequence_length = len(emission_sequence)
            
            emission_state_sets = []
            for symbol in emission_sequence:
                emission_state_sets.append(find_emission_states(symbol, emission_matrix, emission_symbols, len(state_symbols)))
            
            cartesian_product_states = list(itertools.product(*emission_state_sets))
            
            for transition in cartesian_product_states:
                if is_valid_transition([state_symbols[state] for state in transition], transition_matrix, state_symbols, initial_probabilities):
                    possible_transitions.append(transition)
            
            state_probability_pairs = []
            max_probability_sequence = ('', 0.0)
            
            for path in possible_transitions:
                probability = calculate_sequence_probability([state_symbols[state] for state in path], emission_sequence, state_symbols, emission_symbols, transition_matrix, emission_matrix, initial_probabilities)
                state_sequence = [state_symbols[state] for state in path]
                state_probability_pairs.append((state_sequence, probability))
                print("Next probable sequence is:", state_sequence, "with probability:", probability)
                
                if max_probability_sequence[1] < probability:
                    max_probability_sequence = (state_sequence, probability)
            
            print("Maximum probability sequence is:", max_probability_sequence[0], "with probability:", max_probability_sequence[1])
        else:
            print("Invalid emission sequence. Please enter a valid sequence.")
   
# Call the function
hmm_path_finder()
