import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

# Define the directory path
directory_path = 'summaries'

# Initialize data structures
group_data = {}
grouphard_data = {}

# Function to load data from CSV with column names in the second row
def load_data(file_path):
    # Read the second row as header
    df = pd.read_csv(file_path, skiprows=1)
    return df

# Read and separate files into Group and GroupHard categories
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        category, base_name = filename.split('_', 1)
        file_path = os.path.join(directory_path, filename)

        # Load data with adjusted header
        data = load_data(file_path)

        if category == 'Group':
            group_data[base_name] = data
        elif category == 'GroupHard':
            grouphard_data[base_name] = data

# Define custom color mapping
perplexity_color_pairs = {
    'perplexity0_prime': 'r', 
    'perplexity1': 'orange', 'perplexity1_prime': 'orange',
    'perplexity2': 'green', 'perplexity2_prime': 'green',
    'perplexity3': 'm', 'perplexity3_prime': 'm',
    'perplexity4': 'b', 'perplexity4_prime': 'b',
    'perplexity5': 'yellow'
}

# Combine data for matching files and plot
for base_name in group_data.keys():
    # if 'gpt-j' not in base_name or 'mcf' not in base_name or 'NEIGHBOR' in base_name:
    #     continue
    if base_name in grouphard_data:
        print(base_name)
        group_df = group_data[base_name]
        # print(group_df)
        grouphard_df = grouphard_data[base_name]
        # print(grouphard_df)

        x_labels = []
        bar_data = {'Group': [], 'GroupHard': [], 'm-ppl-50': []}
        bar_colors = []

        for index, (g_row, gh_row) in enumerate(zip(group_df.values, grouphard_df.values)):
            if index == 0:
                x_labels.append(f'z{index}_prime')
                bar_data['Group'].append(g_row[10])  # init (Group)
                bar_data['GroupHard'].append(gh_row[10])  # init (GroupHard)
                bar_data['m-ppl-50'].append(g_row[9])  # m-ppl-50
                bar_colors.append(perplexity_color_pairs.get(f'perplexity{index}_prime', 'gray'))
            else:
                for col_idx, col_name in enumerate(['init', 'final']):
                    if index == 1 and col_name == 'init':
                        continue
                    suffix = '' if col_name == 'final' else '_prime'
                    x_label = f'z{index-1}{suffix}' if col_name == 'init' else f'z{index}{suffix}'
                    x_labels.append(x_label)
                    bar_data['Group'].append(g_row[col_idx + 10])  # init/final (Group)
                    bar_data['GroupHard'].append(gh_row[col_idx + 10])  # init/final (GroupHard)
                    bar_data['m-ppl-50'].append(g_row[9] if col_name == 'init' else 0)
                    bar_colors.append(perplexity_color_pairs.get(f'perplexity{index-1}{suffix}' if col_name == 'init' else f'perplexity{index}{suffix}', 'gray'))

        # Plotting the bar graph
        indices = np.arange(len(x_labels))
        # bar_width = 0.25
        bar_width = 0.65

        fig, ax = plt.subplots(figsize=(12, 8))
        # print("Group:", bar_data['Group'])
        # print("GroupHard:", bar_data['GroupHard'])
        # print("m-ppl-50:", bar_data['m-ppl-50'])

        # bars1 = ax.bar(indices - bar_width, bar_data['GroupHard'], bar_width, color=bar_colors, alpha=0.6)
        bars2 = ax.bar(indices, bar_data['Group'], bar_width, color=bar_colors)
        # bars3 = ax.bar(indices + bar_width, bar_data['m-ppl-50'], bar_width, color="grey")

        # Set labels and title
        # ax.set_title(f'Comparison of Average Perplexities for {base_name}')
        ax.set_title("Model Perplexity for 'Microsoft' given 'iPhone developed by' as input", fontsize=16)
        # ax.set_xlabel('z Keys')
        # ax.set_ylabel('Avg Perplexity (log)')
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_yscale('log')

        # Assuming bars1 is your collection of bars
        # for bar in bars1:
        #     bar.set_hatch('//')  # Use '//' for dashed effect, or '|' for vertical lines, etc.
        
        # for bar in bars1:
        #     bar.set_edgecolor('black')  # Set the edge color to black
        #     bar.set_linestyle('--')  # Set the border to dashed lines# Dynamically group bars2 into tuples of at most 2 elements
        # grouped_bars = [(bars2[i], bars2[i + 1]) if i + 1 < len(bars2) else (bars2[i],) 
        #                 for i in range(1, len(bars2), 2)]
        
        # Dynamically group bars2 into tuples of at most 2 elements and take only the first element
        grouped_bars = [bars2[i] for i in range(1, len(bars2), 2)]

        # Create dynamic labels based on iterations
        labels = [f"Iteration {i + 1}" for i in range(len(grouped_bars))]


        
        # Custom legend with handler for tuple-based grouping
        common_heading = "Perplexity(Microsoft | iPhone developed by)"
        legend = ax.legend(
            # [(*bars1,), (*bars2,), (*bars3,)],
            # ['Hard Cases', 'All Cases', 'M_PPL_50'],
            # [(*bars2,), (*bars3,)],
            # [(bars2[1],bars2[2]),(bars2[3],bars2[4]),(bars2[5],bars2[6]),(bars2[7],bars2[8]),(bars2[9])],
            # ['All Cases', 'M_PPL_50'],
            # ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5'],
            grouped_bars,
            labels,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=16,
            title=common_heading,
        )
        # Update the font size of the legend title
        legend.set_title(common_heading)
        legend.get_title().set_fontsize(16)  # Adjust the font size as needed

        # Extract the base numbers from the z-labels
        def get_base_number(label):
            # Remove 'z' prefix and extract the number
            num = label.replace('z', '').split('_')[0]
            return int(num)

        # Group the labels by their base numbers
        label_groups = {}
        for label in x_labels:
            base_num = get_base_number(label)
            if base_num not in label_groups:
                label_groups[base_num] = []
            label_groups[base_num].append(label)

        # Calculate the positions for the group labels
        group_positions = {}
        for base_num in label_groups:
            indices_in_group = [i for i, label in enumerate(x_labels) if get_base_number(label) == base_num]
            group_positions[base_num] = sum(indices_in_group) / len(indices_in_group)
    

        ax.set_xticks(indices)
        new_x_labels = []
        for label in x_labels:
            if label == 'z0_prime':
                new_x_labels.append('Unedited')
            elif 'prime' in label:
                new_x_labels.append('S2:Spread')
            else:
                new_x_labels.append('S1:Optimization')
        ax.set_xticklabels(new_x_labels, rotation=45, ha='right', fontsize=12)

        # # Create a second x-axis for the group numbers
        # ax2 = ax.twiny()
        # ax2.spines['top'].set_visible(False)
        
        # ax2.set_xlim(ax.get_xlim())
        # ax2.set_xticks(list(group_positions.values()))
        # # ax2.set_xticklabels([f'{base_name.split("_")[0]}x'+str(num) for num in group_positions.keys()])
        # ax2.set_xticklabels([f'Iteration '+str(num) for num in group_positions.keys()])
        # ax2.tick_params(axis='x', pad=9, labelsize=12)  # Add some padding between the two sets of labels

        # # Move the second x-axis to the bottom
        # ax2.spines['bottom'].set_position(('outward', 51))
        # ax2.xaxis.set_ticks_position('bottom')

        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.15)

        # Save the plot as a PDF in the summaries directory
        output_path = os.path.join(directory_path, f'{base_name}.pdf')
        plt.savefig(output_path)

        plt.close(fig)  # Close the figure after saving to free memory
