import os
import json
from pathlib import Path

def load_results(file_path):
    """ Load JSON results from a given file path. """
    with open(file_path, 'r') as file:
        return json.load(file)

def save_results(data, save_path):
    """ Save JSON data to a given file path. """
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_algo_names_with_neighbor(base_dir):
    """ Identify all ALGO names with '_NEIGHBOR' in their name, but exclude '_WO_NEIGHBOR'. """
    algo_names = set()
    for entry in os.listdir(base_dir):
        if (
            os.path.isdir(os.path.join(base_dir, entry)) and
            entry.endswith('_NEIGHBOR') and
            not entry.endswith('_WO_NEIGHBOR')
        ):
            # Remove the '_NEIGHBOR' suffix to get the base ALGO name
            algo_name = entry[:-9]  # Remove '_NEIGHBOR'
            algo_names.add(algo_name)
    return algo_names

def filter_neighborhood_data(case, neighbor_prompts):
    """ Filter case's neighborhood data if prompts don't match neighbor prompts. """
    filtered_indices = [
        i for i, prompt in enumerate(case['requested_rewrite']['locality']['neighborhood']['prompt'])
        if prompt in neighbor_prompts
    ]
    
    # If no indices match, return None to indicate removal of this case
    if not filtered_indices:
        return None

    # Retain only matched indices
    def filter_by_indices(data, indices):
        return [data[i] for i in indices]

    # Update neighborhood-related fields based on filtered indices
    case['requested_rewrite']['locality']['neighborhood']['prompt'] = filter_by_indices(
        case['requested_rewrite']['locality']['neighborhood']['prompt'], filtered_indices
    )
    case['requested_rewrite']['locality']['neighborhood']['ground_truth'] = filter_by_indices(
        case['requested_rewrite']['locality']['neighborhood']['ground_truth'], filtered_indices
    )
    case['pre']['locality']['neighborhood_output'] = filter_by_indices(
        case['pre']['locality']['neighborhood_output'], filtered_indices
    )
    case['pre']['locality']['neighborhood_success'] = filter_by_indices(
        case['pre']['locality']['neighborhood_success'], filtered_indices
    )
    case['post']['locality']['neighborhood_output'] = filter_by_indices(
        case['post']['locality']['neighborhood_output'], filtered_indices
    )
    case['post']['locality']['neighborhood_success'] = filter_by_indices(
        case['post']['locality']['neighborhood_success'], filtered_indices
    )
    case['post']['locality']['neighborhood_acc'] = filter_by_indices(
        case['post']['locality']['neighborhood_acc'], filtered_indices
    )

    return case

def extract_matching_results(base_dir):
    # Detect all ALGO names that have corresponding '_NEIGHBOR' directories
    algo_names = get_algo_names_with_neighbor(base_dir)

    # Process each detected ALGO
    for algo_name in algo_names:
        algo_dir = os.path.join(base_dir, algo_name)
        neighbor_dir = os.path.join(base_dir, f"{algo_name}_NEIGHBOR")
        output_dir = os.path.join(base_dir, f"{algo_name}_WO_NEIGHBOR")

        if not os.path.exists(algo_dir) or not os.path.exists(neighbor_dir):
            print(f"Directories {algo_dir} or {neighbor_dir} do not exist.")
            continue

        # Traverse the ALGO_NAME_NEIGHBOR directory to collect prompts
        neighbor_prompts = set()
        for root, _, files in os.walk(neighbor_dir):
            for file_name in files:
                if file_name.startswith('results_') and file_name.endswith('.json'):
                    file_path = Path(root) / file_name
                    neighbor_results = load_results(file_path)

                    # Collect all prompts from NEIGHBOR results
                    for case in neighbor_results:
                        prompt = case['requested_rewrite']['prompt']
                        neighbor_prompts.add(prompt)
                        neighbor_prompts.update(case['requested_rewrite']['locality']['neighborhood']['prompt'])

        # Traverse the ALGO_NAME directory and filter results based on matching prompts
        for root, _, files in os.walk(algo_dir):
            for file_name in files:
                if file_name.startswith('results_') and file_name.endswith('.json'):
                    file_path = Path(root) / file_name
                    algo_results = load_results(file_path)

                    # Filter results based on prompts found in NEIGHBOR directory
                    filtered_results = []
                    for case in algo_results:
                        # Check if the main prompt is in neighbor prompts
                        if case['requested_rewrite']['prompt'] in neighbor_prompts:
                            # Also filter the neighborhood prompts
                            filtered_case = filter_neighborhood_data(case, neighbor_prompts)
                            if filtered_case:
                                filtered_results.append(filtered_case)

                    # Skip saving if no matching results found
                    if not filtered_results:
                        continue

                    # Construct the new save path with modified directory
                    relative_path = Path(root).relative_to(algo_dir)
                    save_path = Path(output_dir) / relative_path / file_name
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the filtered results
                    save_results(filtered_results, save_path)

if __name__ == "__main__":
    # Specify the base directory containing the ALGO and ALGO_NEIGHBOR directories
    base_directory = "./results"
    extract_matching_results(base_directory)
