import json
import os

def combine_json_files(count_list, base_dir):
    combined_data = []
    total_count = sum(count_list)
    case_id = 0  # Initialize case_id counter
    
    # Loop through each count in the list
    for C in count_list:
        directory = os.path.join(base_dir, f"mcf_random_subsets_{C}")
        file_path = os.path.join(directory, f"random_subsets_{C}.json")
        
        # Check if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                for record in data:
                    record["case_id"] = case_id  # Add the case_id field
                    case_id += 1  # Increment the case_id
                combined_data.extend(data)  # Add the contents of the file to the combined list
        else:
            print(f"Warning: {file_path} not found")
    
    # Save the combined data to a new JSON file
    output_directory = os.path.join(base_dir, f"mcf_random_subsets_{total_count}")
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, f"random_subset_{total_count}.json")
    
    with open(output_file_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)
    
    print(f"Combined JSON saved to: {output_file_path}")

# Example usage
count_list = [317, 318, 325]  # List of counts, e.g., [C1, C2, ...]
base_directory = './data'  # Path to the directory containing the data directories

combine_json_files(count_list, base_directory)