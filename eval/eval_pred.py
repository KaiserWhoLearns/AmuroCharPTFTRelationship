import pdb
import json
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import bootstrap
from evaluate import evaluator, load

GENERATION_TASKS = {'xsum', 'xlsum', 'cnn', 'socialiqa', 'sciq', 'tweetqa'}

def evaluate_pred(pred_file_name, dataset_name='xsum', bs=True):
    # Load gold jsonl
    # Load predictons
    with open(pred_file_name, "r") as f:
        instances = [json.loads(x) for x in f.readlines()]

    # Strip gold sequences and predictions
    preds = []
    golds = []
    for instance in instances:
        if 'completion' in instance:
            preds.append(instance["output"])
            golds.append(instance["completion"])
        else:
            # Chat format
            preds.append(instance["output"])
            golds.append(instance['messages'][1]['content'])
    # pdb.set_trace()
    # Put into evaluation
    if dataset_name in GENERATION_TASKS or dataset_name.split('_')[0] in GENERATION_TASKS:
        # Throw the two lists into metric computation
        if bs:
            metric = load('rouge', keep_in_memory=True)
            list_of_rouges = metric.compute(predictions=preds, references=golds, rouge_types=['rougeL'], use_aggregator=False)['rougeL']
            result = bootstrap((list_of_rouges,), np.average, confidence_level=0.95, method='percentile', n_resamples=3000)
            results = dict()
            results['rougeL'] = (result.confidence_interval.high - result.confidence_interval.low) / 2 + result.confidence_interval.low
            results['std'] = result.standard_error
        else:
            metric = load('rouge', keep_in_memory=True, rouge_types=['rougeL'])
            results = metric.compute(predictions=preds, references=golds)
    else:
        # Compute accuracy
        metric = load('accuracy', keep_in_memory=True)
        if 'llmbar' not in dataset_name:
            preds = [strip_labels(pred, dataset_name) for pred in preds]
            # Process golds
            golds = remap_golds(golds)
        # Per-instance computation and bootsraping
        results = metric.compute(predictions=preds, references=golds)
        if bs:
            accs = [golds[i] == preds[i] for i in range(len(golds))]
            result = bootstrap((accs,), np.average, confidence_level=0.95, method='percentile', n_resamples=3000)
            results = dict()
            results['accuracy'] = (result.confidence_interval.high - result.confidence_interval.low) / 2 + result.confidence_interval.low
            results['std'] = result.standard_error
    print(results)
    return results

def strip_labels(pred, dataset_name):
    """
    Strip the label for classification tasks
    """
    # Strip the predictions
    # NLI datasets
    if 'nli' in dataset_name or 'rte' in dataset_name:
        if 'contradiction' in pred:
            return 2
        elif 'neutral' in pred:
            return 1
        elif 'entailment' in pred:
            return 0
    elif 'llmbar' in dataset_name:
        if '1' in pred:
            return 1
        elif '2' in pred:
            return 2
    else:
        # Paraphrase Detection datasets
        if 'no' in pred:
            return 0
        elif 'yes' in pred:
            return 1
    return -1

def remap_golds(golds):
    text_to_label = {'entailment': 0, 'neutral': 1, 'contradiction': 2, 
    ' entailment': 0, ' neutral': 1, ' contradiction': 2, 
    'yes': 1, 'no': 0, ' yes': 1, ' no': 0}
    return [text_to_label[gold] for gold in golds]

if __name__ == "__main__":

    base_dir = os.getenv("base_dir")
    output_file_name = os.getenv("output_file_name")
    evaluate_pred(os.path.join(base_dir, 'output', 'predictions', output_file_name +'.jsonl'), dataset_name='socialiqa')

    # Evaluate for hyperparameter tuning
    # evaluate_pred(os.path.join(base_dir, 'output', 'hp_tuning', 'predictions', output_file_name +'.jsonl'), dataset_name='socialiqa')