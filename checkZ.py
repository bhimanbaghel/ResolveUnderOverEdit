from easyeditor import BaseEditor
from easyeditor import MEMITRecursiveHyperParams, MEMITRecursiveSpreadHyperParams, PMETRecursiveHyperParams, ROMERecursiveHyperParams, MEMITRecursiveNeighborHyperParams, PMETRecursiveNeighborHyperParams, AlphaEditRecursiveHyperParams, AlphaEditRecursiveNeighborHyperParams
import argparse
import os
import json
import numpy as np

def getData(
    ds_name: str,
    model_name: str,
    num_edits: int = 10,
    ds_subset: int = 0,    
):
    dataPath = os.path.join('.','data', f'{ds_name}_random_subsets_{num_edits}', f'random_subset_{ds_subset}.json')
    with open(dataPath) as f:
        requests = json.load(f)
    
    case_ids, prompts, ground_truth, target_new, subject, rephrase_prompts, neighborhood_samples = [], [], [], [], [], [], [] 
    locality_inputs = {
        'neighborhood':{
            'prompt': [],
            'ground_truth': []
        },
    }

    if ds_name == 'mcf':
        for request in requests:
            case_ids.append(request['case_id'])
            try:
                prompts.append(request['requested_rewrite']['prompt'].format(request['requested_rewrite']['subject']))
            except KeyError:
                prompts.append(request["prompt"])
            try:
                ground_truth.append(request['requested_rewrite']['target_true']['str'])
            except KeyError:
                ground_truth.append(request["ground_truth"])
            try:
                target_new.append(request['requested_rewrite']['target_new']['str'])
            except KeyError:
                target_new.append(request["target_new"])
            try:
                subject.append(request['requested_rewrite']['subject'])
            except KeyError:
                subject.append(request["subject"])
            try:
                rephrase_prompts.append(request['paraphrase_prompts'])
            except KeyError:
                rephrase_prompts.append(request['rephrase_prompt'])
            if 'locality' in request:
                locality_inputs["neighborhood"]['prompt'].append([neighbor for neighbor in request['locality']['neighborhood']['prompt']])
                locality_inputs['neighborhood']['ground_truth'].append([neighbor for neighbor in request['locality']['neighborhood']['ground_truth']])
            else:
                locality_inputs['neighborhood']['prompt'].append([neighbor for neighbor in request['neighborhood_prompts']])
                locality_inputs['neighborhood']['ground_truth'].append([request['requested_rewrite']['target_true']['str'] for _ in request['neighborhood_prompts']])
            neighborhood_samples.append(request["neighborhood_samples"])
    elif ds_name == 'zsre': 
        for case_id, request in enumerate(requests):
            prompts.append(request)
            subject.append(request["subject"])
            neighborhood_samples.append(request["neighborhood_samples"])
    
    return case_ids, prompts, ground_truth, target_new, subject, locality_inputs, rephrase_prompts, neighborhood_samples


def main(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    num_edits: int = 10,
    ds_subset: int = 0,
    iterations: int = 3,
    summarize: bool = False
):
    if "_NEIGHBOR" in alg_name and ds_name != "mcf":
        print("Neighbor experiment only work for Counterfact dataset. Pass mcf as ds_name")
        return
    if summarize:
        summaries = []
        summaries_false_case = []
        def get_all_acc_success_keys(dict_list):
            all_keys = set()

            def recursive_keys(d):
                for k, v in d.items():
                    if k.endswith('acc') or k.endswith('success'):
                        all_keys.add(k)
                    if isinstance(v, dict):
                        recursive_keys(v)
                        
            for dictionary in dict_list:
                recursive_keys(dictionary)

            return all_keys
        
        logs_dir = os.path.join('./', 'results', alg_name, model_name, ds_name, f'N_{num_edits}', f'S_{ds_subset}', f'I_{iterations}')
        for i_r in range(iterations):
            result_file = os.path.join(logs_dir, f'results_{i_r+1}.json')
            
            with open(result_file, 'r') as file:
                all_metrics = json.load(file)
            
            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc", "rewrite_success", "rephrase_success", 'rewrite_ppl', 'init_ppl', 'final_ppl']:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in get_all_acc_success_keys(all_metrics):
                            metrics = [np.mean(metric[eval][key][lkey]) for metric in all_metrics if lkey in metric[eval][key].keys()]
                            if len(metrics) > 0:
                                mean_metrics[eval][key][lkey] = round(np.mean(metrics), 2)
            summaries.append(mean_metrics)
    else:
        if alg_name == "MEMIT_RECURSIVE":
            hparams = MEMITRecursiveHyperParams.from_hparams(hparams_fname)
        elif alg_name == "MEMIT_RECURSIVE_SPREAD":
            hparams = MEMITRecursiveSpreadHyperParams.from_hparams(hparams_fname)
        elif alg_name == "PMET_RECURSIVE":
            hparams = PMETRecursiveHyperParams.from_hparams(hparams_fname)
        elif alg_name == "ROME_RECURSIVE":
            hparams = ROMERecursiveHyperParams.from_hparams(hparams_fname)
        elif alg_name == "MEMIT_RECURSIVE_NEIGHBOR":
            hparams = MEMITRecursiveNeighborHyperParams.from_hparams(hparams_fname)
        elif alg_name == "PMET_RECURSIVE_NEIGHBOR":
            hparams = PMETRecursiveNeighborHyperParams.from_hparams(hparams_fname)
        elif alg_name == "AlphaEdit_RECURSIVE":
            hparams = AlphaEditRecursiveHyperParams.from_hparams(hparams_fname)
        elif alg_name == "AlphaEdit_RECURSIVE_NEIGHBOR":
            hparams = AlphaEditRecursiveNeighborHyperParams.from_hparams(hparams_fname)
        else:
            print(f"Unsupported Algo:{alg_name}")
            return
        
        case_ids, prompts, ground_truth, target_new, subject, locality_inputs, rephrase_prompts, neighborhood_samples = getData(
            ds_name=ds_name,
            model_name=hparams.model_name,
            num_edits=num_edits,
            ds_subset=ds_subset,
        )
        
        hparams.batch_size = num_edits
        editor = BaseEditor.from_hparams(hparams)
        if "_SPREAD" in alg_name:
            summaries, summaries_false_case, edited_model, _ = editor.batch_edit_recursive_spread(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                subject=subject,
                locality_inputs=locality_inputs,
                rephrase_prompts=rephrase_prompts,
                keep_original_weight=True,
                iterations=iterations,
                ds_name=ds_name,
                num_edits=num_edits,
                ds_subset=ds_subset,
                neighborhood_samples=neighborhood_samples,
            )
        else:
            summaries, summaries_false_case, edited_model, _ = editor.batch_edit_recursive(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                subject=subject,
                locality_inputs=locality_inputs,
                rephrase_prompts=rephrase_prompts,
                keep_original_weight=True,
                iterations=iterations,
                ds_name=ds_name,
                num_edits=num_edits,
                ds_subset=ds_subset,
                neighborhood_samples=neighborhood_samples,
            )
    logs_dir = os.path.join('./', 'results', alg_name, model_name, ds_name, f'N_{num_edits}', f'S_{ds_subset}', f'I_{iterations}')
    os.makedirs(logs_dir, exist_ok=True)
    output_file = os.path.join(logs_dir, 'summary.json')
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=4)
    for summary in summaries:
        print(summary)
    
    if len(summaries_false_case) > 0:
        output_file = os.path.join(logs_dir, 'summaryHard.json')
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(summaries_false_case, f, ensure_ascii=False, indent=4)
        for summary in summaries_false_case:
            print(summary)
    else:
        print("No false cases")
    # print(edited_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT_RECURSIVE", "MEMIT_RECURSIVE_SPREAD", "PMET_RECURSIVE", "ROME_RECURSIVE", "MEMIT_RECURSIVE_NEIGHBOR", "PMET_RECURSIVE_NEIGHBOR", "MEMIT_RECURSIVE_WO_NEIGHBOR", "PMET_RECURSIVE_WO_NEIGHBOR", "AlphaEdit_RECURSIVE", "AlphaEdit_RECURSIVE_NEIGHBOR"],
        default="MEMIT_RECURSIVE",
        help="Editing Algo",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-xl","gpt-j-6B", "llama-2-7b", "llama-3-8b"],
        default="llama-2-7b",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="./hparams/MEMIT_RECURSIVE/llama-7b.yaml",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        choices=["mcf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either MultiCounterFact (mcf), or zsRE (zsre).",
        required=True,
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=10,
        help="Number of rewrites to perform simultaneously.",
        required=True
    )
    parser.add_argument(
        "--ds_subset",
        type=int,
        default="0",
        help="Dataset subset",
        required=True,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default="3",
        help="Dataset subset",
        required=True,
    )
    parser.add_argument(
        "--summarize",
        action='store_true'
    )
    args = parser.parse_args()
    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.num_edits,
        args.ds_subset,
        args.iterations,
        True if args.summarize else False,
    )