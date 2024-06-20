# File to generate performance table for analysis and plotting
import os
import pdb
import glob
import json
import pandas as pd
import sys
sys.path.append(os.environ['base_dir'])
from eval.eval_pred import evaluate_pred

def create_full_perf_table():
    """
    Generate a table that has model_id, dataset, and performance
    """
    # Loop through the prediction folder
    # TODO: Add a functionality to only update the non-overwritten results
    pred_files = glob.glob(os.path.join(os.environ['base_dir'], "output", "predictions", "*.jsonl"))
    perf_table = []
    for pred_file in pred_files:
        # Parse for the model name and dataset name
        pred_file_name = pred_file.split('/')[-1]
        if 'instruct' in pred_file_name or 'inputoutput' in pred_file_name:
            # Parse for instruction format
            if pred_file_name[:4] == 'mnli' or pred_file_name[:6] == 'llmbar' and 'Natural' in pred_file_name:
                # MNLI_Matched and unmatched
                ds_name = "_".join(pred_file_name.split('_')[:3])
                model_name = "_".join(pred_file_name.split('_')[3:])
            elif pred_file_name[:6] == 'llmbar':
                ds_name = "_".join(pred_file_name.split('_')[:4])
                model_name = "_".join(pred_file_name.split('_')[4:])
            else:
                ds_name = "_".join(pred_file_name.split('_')[:2])
                model_name = "_".join(pred_file_name.split('_')[2:])
        elif pred_file_name[:4] == 'mnli' or pred_file_name[:6] == 'llmbar' and 'Natural' in pred_file_name:
            # MNLI_Matched and unmatched
            ds_name = "_".join(pred_file_name.split('_')[:2])
            model_name = "_".join(pred_file_name.split('_')[2:])
        elif pred_file_name[:6] == 'llmbar':
            ds_name = "_".join(pred_file_name.split('_')[:3])
            model_name = "_".join(pred_file_name.split('_')[3:])
        else:
            ds_name = pred_file_name.split('_')[0]
            model_name = "_".join(pred_file_name.split('_')[1:])
        # Remove the .jsonl in model_name
        if '.jsonl' in model_name:
            model_name = model_name.replace(".jsonl", "")
        
        # Compute the performance
        perf_res = evaluate_pred(pred_file, dataset_name=ds_name)
        # Parse for corresponding performance slot
        perf = perf_res['accuracy'] if 'accuracy' in perf_res else perf_res['rougeL']
        # Add the model_id and dataset performance to table
        perf_table.append({
            "model_id": model_name,
            "eval dataset": ds_name,
            "Performance": perf,
            'std': perf_res['std'] 
        })

    # Save table
    pd.DataFrame(perf_table).to_csv(os.path.join(os.environ['base_dir'], 'results', 'analysis', 'all_perf_table.csv'))
    
def gather_instruction_tuning_performance():
    # Directly loads performance from the eval_results outputed by the eval script
    perf_table = []
    # For each dataset, load correspoding performance sheet
    for ds_name in ['alpaca', 'gsm', 'mmlu', 'truthfulqa', 'toxigen']:
        perf_tab = helper_load_it_perf_by_ds(ds_name)
        perf_table += perf_tab

    pd.DataFrame(perf_table).to_csv(os.path.join(os.environ['base_dir'], 'results', 'analysis', 'it_perf_table.csv'))

def helper_load_it_perf_by_ds(ds_name):
    instances = []
    PERF_KEYS = {"mmlu": "average_acc", "toxigen": "overall",
                 "gsm": "exact_match", "alpaca": "win_rate", "truthfulqa": "truth-info acc"}
    # Recursively load dir from eval_results dir
    pred_folders = glob.glob(os.path.join(os.environ['base_dir'], "eval_results", ds_name, "*"), recursive=True)
    for pred_folder in pred_folders:
        model_name = pred_folder.split('/')[-1]
        # Load the overall pred
        try:
            # Sometimes the file has not been written yet
            pred_file = open(os.path.join(pred_folder, 'metrics.json'))
            perf = json.load(pred_file)
            if ds_name == "alpaca":
                instances.append({
                    "model_id": model_name,
                    "eval dataset": ds_name,
                    "Performance": perf[PERF_KEYS[ds_name]][model_name + "-greedy-long"]
                })
            else:
                instances.append({
                    "model_id": model_name,
                    "eval dataset": ds_name,
                    "Performance": perf[PERF_KEYS[ds_name]]
                })
            pred_file.close()
        except:
            pass
    return instances


def gather_official_instruction_tuning_performance():
    # Directly loads performance from the metrics.json outputed by the eval script
    perf_table = []
    # For each dataset, load correspoding performance sheet
    pred_files = glob.glob(os.path.join(os.environ['base_dir'], "eval_results", "eval_pipeline", "*"), recursive=True)
    for file in pred_files:
        if 'arc' in file:
            # There exist a '_' in the dataset name
            dataset_name = "_".join(file.split('/')[-1].split('_')[0:2])
        else:
            dataset_name = file.split('/')[-1].split('_')[0]
        # Get the model_name
        model_name = "".join(file.split('/')[-1].split('.json')[0][len(dataset_name)+1:])
        pred_file = open(file)
        perf = json.load(pred_file)['metrics']
        pred_file.close()
        perf_table.append(
            {
                    "model_id": model_name,
                    "eval dataset": dataset_name,
                    "Performance": perf[0]['metrics']['rc_metrics']['acc']
                }
        )

    pd.DataFrame(perf_table).to_csv(os.path.join(os.environ['base_dir'], 'results', 'analysis', 'official_eval_table.csv'))


if __name__ == "__main__":
    create_full_perf_table()
    # gather_instruction_tuning_performance()
    gather_official_instruction_tuning_performance()