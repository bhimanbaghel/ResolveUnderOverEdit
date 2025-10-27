import os
import pandas as pd
from glob import glob
from collections import defaultdict
from scipy.stats import hmean
import json

def compute_harmonic_mean(values):
    """Compute harmonic mean, avoiding division by zero."""
    values = [v for v in values if v > 0]
    return hmean(values) if values else 0

def process_summary_file(file_path):
    with open(file_path, 'r') as f:
        # Load the JSON data properly
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
            return []

    results = []

    pre_count=0

    for record in data:
        # Ensure each record is a dictionary
        if isinstance(record, str):
            try:
                record = json.loads(record)
            except json.JSONDecodeError:
                print(f"Error decoding JSON record in file: {file_path}")
                continue
        
        post = record.get('post', {})
        
        # Efficacy metrics
        efficacy_acc = post.get('rewrite_acc', None)
        efficacy_success = post.get('rewrite_success', None)
        
        # Generalization metrics
        generalization_acc = post.get('rephrase_acc', None)
        generalization_success = post.get('rephrase_success', None)
        
        # Specificity metrics
        specificity_acc = post.get('locality', {}).get('neighborhood_acc', None)
        specificity_success = post.get('locality', {}).get('neighborhood_success', None)
        
        # Compute the harmonic mean for accuracy and success
        score_acc = compute_harmonic_mean([efficacy_acc, generalization_acc, specificity_acc])
        score_success = compute_harmonic_mean([efficacy_success, generalization_success, specificity_success])

        # Perplexity metrics
        m_ppl_50 = record.get('m_ppl', None)
        init_ppl = post.get('init_ppl', None)
        _init_ppl = post.get('init_ppl_verify', None)
        final_ppl = post.get('final_ppl', None)
        rewrite_ppl = post.get('rewrite_ppl', None)
        if rewrite_ppl != None or final_ppl != None:
            approximation_error = rewrite_ppl - final_ppl
        else:
            approximation_error = None
        
        if pre_count==0:
            pre_m_ppl_50 = record.get('m_ppl_pre')
            pre = record.get('pre', {})
            pre_efficacy_acc = pre.get('rewrite_acc', None)
            pre_efficacy_success = pre.get('rewrite_success', None)
            pre_generalization_acc = pre.get('rephrase_acc', None)
            pre_generalization_success = pre.get('rephrase_success', None)
            pre_specificity_acc = pre.get('locality', {}).get('neighborhood_acc', None)
            pre_specificity_success = pre.get('locality', {}).get('neighborhood_success', None)
            pre_score_acc = None
            pre_score_success = compute_harmonic_mean([pre_efficacy_success, pre_generalization_success, pre_specificity_success])
            results.append({
                'efficacy_acc': pre_efficacy_acc,
                'efficacy_success': pre_efficacy_success,
                'generalization_acc': pre_generalization_acc,
                'generalization_success': pre_generalization_success,
                'specificity_acc': pre_specificity_acc,
                'specificity_success': pre_specificity_success,
                'score_acc': pre_score_acc,
                'score_success': pre_score_success,
                'perplexity_m-ppl-50':pre_m_ppl_50,
                'perplexity_init':init_ppl,
                'perplexity_init_verify':None,
                'perplexity_final':None,
                'perplexity_rewrite':None,
                'apx-error_rewrite-final':None,
            })
            pre_count+=1

        results.append({
            'efficacy_acc': efficacy_acc,
            'efficacy_success': efficacy_success,
            'generalization_acc': generalization_acc,
            'generalization_success': generalization_success,
            'specificity_acc': specificity_acc,
            'specificity_success': specificity_success,
            'score_acc': score_acc,
            'score_success': score_success,
            'perplexity_m-ppl-50':m_ppl_50,
            'perplexity_init':init_ppl,
            'perplexity_init_verify':_init_ppl,
            'perplexity_final':final_ppl,
            'perplexity_rewrite':rewrite_ppl,
            'apx-error_rewrite-final':approximation_error

        })
    
    return results

def traverse_results_directory(base_dir, summary):
    results = defaultdict(list)
    summary_files = glob(os.path.join(base_dir, '**', summary), recursive=True)
    
    for summary_file in summary_files:
        # Extracting the directory structure to use as a group key
        path_parts = summary_file.split(os.sep)
        algo, model, dataset, num_examples, subset_id, num_iterations = path_parts[-7:-1]
        group_key = (algo, model, dataset, num_examples, subset_id, num_iterations)

        results[group_key].extend(process_summary_file(summary_file))


    # Convert each group's results into a DataFrame with MultiIndex columns
    for group_key, group_data in results.items():
        df = pd.DataFrame(group_data)
        
        # Create a MultiIndex for columns
        columns = pd.MultiIndex.from_tuples(
            [
                ('Efficacy', 'acc'),
                ('Efficacy', 'success'),
                ('Generalization', 'acc'),
                ('Generalization', 'success'),
                ('Specificity', 'acc'),
                ('Specificity', 'success'),
                ('Score', 'acc'),
                ('Score', 'success'),
                ('Perplexity', 'm-ppl-50'),
                ('Perplexity', 'init'),
                ('Perplexity', 'init_verify'),
                ('Perplexity', 'final'),
                ('Perplexity', 'rewrite'),
                ('ApproxError', 'rewrite - final'),
            ],
            names=['Metric', 'MEMIT']
        )
        
        # Restructure the DataFrame with the MultiIndex columns
        df.columns = columns
        
        # Create a single string identifier for each group
        group_name = "_".join(group_key)
        
        # Print the group name as a heading
        if 'Hard' in summary:
            
            print(f"\nGroupHard: {group_name}")
            group_name = f'GroupHard_{group_name}'
        else:
            print(f"\nGroup: {group_name}")
            group_name = f'Group_{group_name}'
        
        # Drop pre-values (since they're not needed) and print DataFrame
        if not df.empty:
            # df.index = range(1, len(df) + 1)
            print(df)
            df.to_csv(os.path.join('.', 'summaries', f'{group_name}.csv'))
        


# Example usage:
base_directory = './results'  # Replace with your directory path
traverse_results_directory(base_directory, 'summary.json')
traverse_results_directory(base_directory, 'summaryHard.json')
