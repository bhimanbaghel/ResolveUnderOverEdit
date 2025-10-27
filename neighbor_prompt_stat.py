import json
import statistics
import os
import math

def calculate_statistics(json_file):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Extract the number of prompts for each dictionary
    prompt_counts = [len(item["locality"]["neighborhood"]["prompt"]) for item in data]
    
    # Calculate mean, median, and mode
    mean_value = math.ceil(statistics.mean(prompt_counts))
    median_value = statistics.median(prompt_counts)
    
    try:
        mode_value = statistics.mode(prompt_counts)
    except statistics.StatisticsError:
        mode_value = "No unique mode"
    
    return mean_value, median_value, mode_value

# Example usage
N_count = 1340
json_file = os.path.join('data', f'mcf_random_subsets_{N_count}', f'random_subset_{N_count}.json')
mean, median, mode = calculate_statistics(json_file)
print(f"Mean: {mean}, Median: {median}, Mode: {mode}")
