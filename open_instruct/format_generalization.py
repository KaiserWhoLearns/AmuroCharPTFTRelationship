'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
'''
import os
import json
import pdb
from datasets import load_dataset

def process_xsum(output_dir, use_prompt_format=True):
    """
    data source: https://huggingface.co/datasets/EdinburghNLP/xsum
    Should only be ran once
    use_prompt_format = True to output it in prompt completion format
    """
    def format_single_xsum(example):
        example['input'] = '### Input: ' + example['document'] + '\n ### Summary: '
        example['text'] = '### Input: ' + example['document'] + '\n ### Summary: ' + example['summary']
        example['metadata'] = 'xsum'
        return example

    raw_datasets = load_dataset("EdinburghNLP/xsum")
    # Shuffle and sample the training set
    raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42).select(range(6000))
    # Arrange the data format
    full_data = raw_datasets.map(format_single_xsum)
    full_data = full_data.remove_columns('document')
    # full_data = full_data.remove_columns('summary')

    output_path = os.path.join(output_dir, "xsum.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data['train']):
            prompt = example["input"]
            completion = example["summary"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "xsum",
                "id": f"xsum_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "xsum",
                    "id": f"xsum_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_summarization_test_set(output_dir, use_prompt_format=True, instruction_format='instruct'):
    """
    data source: https://huggingface.co/datasets/EdinburghNLP/xsum
    Should only be ran once
    """
    def format_single_xsum(example):
        if instruction_format == 'instruct':
            example['input'] = f'Please read the following text: {example["document"]} Provide a summary: '
            example['text'] = example['input'] + example['summary']
        elif instruction_format == 'inputoutput':
            example['input'] = example["document"]
            example['text'] = example['input'] + example['summary']
        else:
            example['input'] = '### Input: ' + example['document'] + '\n ### Summary: '
            example['text'] = '### Input: ' + example['document'] + '\n ### Summary: ' + example['summary']
        example['metadata'] = 'xsum'
        return example

    raw_datasets = load_dataset("EdinburghNLP/xsum")
    # Arrange the data format
    full_data = raw_datasets.map(format_single_xsum)
    full_data = full_data.remove_columns('document')
    # full_data = full_data.remove_columns('summary')

    if instruction_format == 'instruct' or instruction_format == 'inputoutput':
        output_path = os.path.join(output_dir, f"xsum_{instruction_format}.jsonl")
    else:
        output_path = os.path.join(output_dir, "xsum.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data['test']):
            prompt = example["input"]
            completion = example["summary"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "xsum",
                "id": f"xsum_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "xsum",
                    "id": f"xsum_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

    def format_single_xlsum(example):
        if instruction_format == 'instruct':
            example['input'] = f'Please read the following text: {example["text"]} Provide a summary: '
            example['text'] = example['input'] + example['summary']
        elif instruction_format == 'inputoutput':
            example['input'] = example["text"]
            example['text'] = example['input'] + example['summary']
        else:
            example['input'] = '### Input: ' + example['text'] + '\n ### Summary: '
            example['text'] = '### Input: ' + example['text'] + '\n ### Summary: ' + example['summary']
        example['metadata'] = 'xlsum'
        return example

    raw_datasets = load_dataset("csebuetnlp/xlsum", 'english', split='test')
    # Arrange the data format
    full_data = raw_datasets.map(format_single_xlsum)
    # full_data = full_data.remove_columns('summary')
    if instruction_format == 'instruct' or instruction_format == 'inputoutput':
        output_path = os.path.join(output_dir, f"xlsum_{instruction_format}.jsonl")
    else:
        output_path = os.path.join(output_dir, "xlsum.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["summary"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "xlsum",
                "id": f"xlsum_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "xlsum",
                    "id": f"xlsum_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

    def format_single_cnn(example):
        if instruction_format == 'instruct':
            example['input'] = f'Please read the following text: {example["article"]} Provide a summary: '
            example['text'] = example['input'] + example['highlights']
        else:
            example['input'] = '### Input: ' + example['article'] + '\n ### Summary: '
            example['text'] = '### Input: ' + example['article'] + '\n ### Summary: ' + example['highlights']
        example['metadata'] = 'cnn'
        return example

    raw_datasets = load_dataset("cnn_dailymail", '3.0.0', split='test', ignore_verifications=True)
    # Arrange the data format
    full_data = raw_datasets.map(format_single_cnn)
    full_data = full_data.remove_columns('article')
    # full_data = full_data.remove_columns('highlights')

    if instruction_format == 'instruct':
        output_path = os.path.join(output_dir, "cnn_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "cnn.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["highlights"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "cnn",
                "id": f"cnn_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "cnn",
                    "id": f"cnn_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_socialiqa(output_dir, use_prompt_format=True):
    """
    data source: https://huggingface.co/datasets/EdinburghNLP/xsum
    Should only be ran once
    """
    def format_single_socialiqa(example):
        example['input'] = '### Input: ' + example['context'] + '\n ### Answer: Answer A: ' + example['answerA'] + ' Answer B: ' + example['answerB'] + 'Answer C: ' + example['answerC'] + '\n ### Question: '
        example['text'] = '### Input: ' + example['context'] + '\n ### Answer: Answer A: ' + example['answerA'] + ' Answer B: ' + example['answerB'] + 'Answer C: ' + example['answerC'] + '\n ### Question: ' +  example['question']
        example['metadata'] = 'socialiqa'
        return example

    raw_datasets = load_dataset("social_i_qa")
    # Shuffle and sample the training set
    raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42).select(range(6000))
    # Arrange the data format
    full_data = raw_datasets.map(format_single_socialiqa)
    full_data = full_data.remove_columns('context')
    full_data = full_data.remove_columns('answerA')
    full_data = full_data.remove_columns('answerB')
    full_data = full_data.remove_columns('answerC')
    # full_data = full_data.remove_columns('question')
    full_data = full_data.remove_columns('label')

    output_path = os.path.join(output_dir, "socialiqa.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data['train']):
            prompt = example["input"]
            completion = example["question"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "socialiqa",
                "id": f"socialiqa_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "socialiqa",
                    "id": f"socialiqa_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_socialiqa_test(output_dir, use_prompt_format=True, instruction_format='instruct'):
    """
    data source: https://huggingface.co/datasets/EdinburghNLP/xsum
    Should only be ran once
    """
    def format_single_socialiqa(example):
        if instruction_format == 'instruct':
            example['input'] = "Given the context: " + example['context'] + " And the answer: Answer A: " + example['answerA'] + " Answer B: " + example['answerB'] + "Answer C: " + example['answerC'] + "Generate a suitable question: "
            example['text'] = example['input'] + example['question']
        elif instruction_format == 'inputoutput':
            example['input'] = example['context'] + example['answerA'] + example['answerB'] + example['answerC']
            example['text'] = example['input'] + example['question']
        else:
            example['input'] = '### Input: ' + example['context'] + '\n ### Answer: Answer A: ' + example['answerA'] + ' Answer B: ' + example['answerB'] + 'Answer C: ' + example['answerC'] + '\n ### Question: '
            example['text'] = '### Input: ' + example['context'] + '\n ### Answer: Answer A: ' + example['answerA'] + ' Answer B: ' + example['answerB'] + 'Answer C: ' + example['answerC'] + '\n ### Question: ' +  example['question']
        example['metadata'] = 'socialiqa'
        return example

    raw_datasets = load_dataset("social_i_qa")
    # Shuffle and sample the training set
    # raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42).select(range(6000))
    # Arrange the data format
    full_data = raw_datasets.map(format_single_socialiqa)
    full_data = full_data.remove_columns('context')
    full_data = full_data.remove_columns('answerA')
    full_data = full_data.remove_columns('answerB')
    full_data = full_data.remove_columns('answerC')
    # full_data = full_data.remove_columns('question')
    full_data = full_data.remove_columns('label')

    if instruction_format == 'instruct' or instruction_format == 'inputoutput':
        output_path = os.path.join(output_dir, f"socialiqa_{instruction_format}.jsonl")
    else:
        output_path = os.path.join(output_dir, "socialiqa.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data['validation']):
            prompt = example["input"]
            completion = example["question"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "socialiqa",
                "id": f"socialiqa_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "socialiqa",
                    "id": f"socialiqa_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_question_generation_test_sets(output_dir, use_prompt_format=True, instruction_format=False):
    """
    data source: sciq (https://huggingface.co/datasets/sciq), TweetQA (https://huggingface.co/datasets/tweet_qa)
    """
    def format_single_sciq(example):
        if instruction_format:
            example['input'] = "Given the context: " + example['support'] + " And the answer: Answer A: " + example['correct_answer'] + " Answer B: " + example['distractor1'] + "Answer C: " + example['distractor2'] + 'Answer D: ' + example['distractor3'] + "Generate a suitable question: "
            example['text'] = example['input'] + example['question']
        else:
            example['input'] = '### Input: ' + example['support'] + '\n ### Answer: Answer A: ' + example['correct_answer'] + ' Answer B: ' + example['distractor1'] + ' Answer C: ' + example['distractor2'] + 'Answer D: ' + example['distractor3'] + '\n ### Question: '
            example['text'] = '### Input: ' + example['support'] + '\n ### Answer: Answer A: ' + example['correct_answer'] + ' Answer B: ' + example['distractor1'] + ' Answer C: ' + example['distractor2'] + 'Answer D: ' + example['distractor3'] + '\n ### Question: ' +  example['question']
        example['metadata'] = 'sciq'
        return example

    raw_datasets = load_dataset('sciq', split='test')
    # Arrange the data format
    full_data = raw_datasets.map(format_single_sciq)
    full_data = full_data.remove_columns('support')
    full_data = full_data.remove_columns('correct_answer')
    full_data = full_data.remove_columns('distractor1')
    full_data = full_data.remove_columns('distractor2')
    full_data = full_data.remove_columns('distractor3')
    
    # Sanity check
    print(full_data[0])
    if instruction_format:
        output_path = os.path.join(output_dir, "sciq_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "sciq.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["question"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "sciq",
                "id": f"sciq_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "sciq",
                    "id": f"sciq_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

    def format_single_tweetqa(example):
        if instruction_format:
            example['input'] = "Given the context: " + example['Tweet'] + " And the answer: " + example['Answer'][0] + "Generate a suitable question: "
            example['text'] = example['input'] + example['Question']
        else:
            example['input'] = '### Input: ' + example['Tweet'] + '\n ### Answer: ' + example['Answer'][0] + '\n ### Question: '
            example['text'] = '### Input: ' + example['Tweet'] + '\n ### Answer: ' + example['Answer'][0] + '\n ### Question: ' +  example['Question']
        example['metadata'] = 'tweetqa'
        return example

    raw_datasets = load_dataset("tweet_qa", split='train')
    # Arrange the data format
    full_data = raw_datasets.map(format_single_tweetqa)
    full_data = full_data.remove_columns('Tweet')
    full_data = full_data.remove_columns('Answer')
    full_data = full_data.remove_columns('qid')

    # Sanity check
    print(full_data[0])
    if instruction_format:
        output_path = os.path.join(output_dir, "tweetqa_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "tweetqa.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["Question"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "tweetqa",
                "id": f"tweetqa_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "tweetqa",
                    "id": f"tweet_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_mnli(output_dir, use_prompt_format=True, get_test=False, mismatched=False, instruction_format='instruct'):
    """
    data source: https://huggingface.co/datasets/glue/viewer/mnli
    Should only be ran once
    """
    def format_single_mnli(example):
        if example['label'] == 1:
            label = 'neutral'
        elif example['label'] == 0:
            label = 'entailment'
        else:
            label = 'contradiction'
        if instruction_format == 'instruct':
            example['input'] = f"Consider the following texts: Text 1: {example['premise']} Text 2: {example['hypothesis']} The relation is "
            example['text'] = example['input'] + label
        elif instruction_format == 'inputoutput':
            example['input'] = f"{example['premise']} {example['hypothesis']}"
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['premise'] + '\n ### Input_2: ' + example['hypothesis'] + '\n ### Inference: '
            example['text'] = '### Input_1: ' + example['premise'] + '\n ### Input_2: ' + example['hypothesis'] + '\n ### Inference: ' + label
        example['text_label'] = label
        example['metadata'] = 'mnli'
        return example

    if not get_test:
        # Can be solved by streaming, or try downgrading dataset?
        raw_datasets = load_dataset('glue', 'mnli')
        # Shuffle and sample the training set
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42).select(range(6000))
    elif mismatched:
        raw_datasets = load_dataset('glue', 'mnli_mismatched', split='validation')
    else:
        raw_datasets = load_dataset('glue', 'mnli_matched', split='validation')
    
    # Arrange the data format
    full_data = raw_datasets.map(format_single_mnli)
    full_data = full_data.remove_columns('premise')
    full_data = full_data.remove_columns('hypothesis')
    # full_data = full_data.remove_columns('label')
    full_data = full_data.remove_columns('idx')
    # Save the processed dataset to directory
    if get_test:
        write_path = output_dir + '/evaluation'
        if mismatched:
            if instruction_format == 'instruct' or instruction_format == 'inputoutput':
                output_path = os.path.join(write_path, f"mnli_mismatched_{instruction_format}.jsonl")
            else:
                output_path = os.path.join(write_path, "mnli_mismatched.jsonl")
        else:
            if instruction_format == 'instruct' or instruction_format == 'inputoutput':
                output_path = os.path.join(write_path, f"mnli_matched_{instruction_format}.jsonl")
            else:
                output_path = os.path.join(write_path, "mnli_matched.jsonl")
    else:
        write_path = output_dir
        output_path = os.path.join(write_path, "mnli.jsonl")
        full_data = full_data['train']

    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["text_label"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "mnli1",
                "id": f"mnli1_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "mnli1",
                    "id": f"mnli1_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_nli_test(output_dir, use_prompt_format=True, instruction_format=False):
    """
    data source: https://huggingface.co/datasets/glue/viewer/mnli
    Should only be ran once
    """
    def format_single_mnli(example):
        if example['label'] == 1:
            label = 'neutral'
        elif example['label'] == 0:
            label = 'entailment'
        else:
            label = 'contradiction'
        if instruction_format:
            example['input'] = f"Consider the following texts: Text 1: {example['premise']} Text 2: {example['hypothesis']} The relation is "
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['premise'] + '\n ### Input_2: ' + example['hypothesis'] + '\n ### Inference: '
            example['text'] = '### Input_1: ' + example['premise'] + '\n ### Input_2: ' + example['hypothesis'] + '\n ### Inference: ' + label
        example['metadata'] = 'mnli'
        example['text_label'] = label
        return example

    raw_datasets = load_dataset('glue', 'mnli_matched', split='validation')
    
    # Arrange the data format
    full_data = raw_datasets.map(format_single_mnli)
    full_data = full_data.remove_columns('premise')
    full_data = full_data.remove_columns('hypothesis')
    full_data = full_data.remove_columns('label')
    full_data = full_data.remove_columns('idx')

    if instruction_format:
        output_path = os.path.join(output_dir, "mnli_matched_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "mnli_matched.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["text_label"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "mnli1",
                "id": f"mnli1_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "mnli1",
                    "id": f"mnli1_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

    def format_single_rte(example):
        # pdb.set_trace()
        label = 'neutral' if example['label'] == 1 else 'entailment'
        if instruction_format:
            example['input'] = f"Consider the following texts: Text 1: {example['sentence1']} Text 2: {example['sentence2']} The relation is "
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['sentence1'] + '\n ### Input_2: ' + example['sentence2'] + '\n ### Inference: '
            example['text'] = '### Input_1: ' + example['sentence1'] + '\n ### Input_2: ' + example['sentence1'] + '\n ### Inference: ' + label
        example['text_label'] = label
        example['metadata'] = 'rte'
        return example

    # Train set of RTE because hidden test sets
    # pdb.set_trace()
    raw_datasets = load_dataset('glue', 'rte', split='train')
    
    # Arrange the data format
    full_data = raw_datasets.map(format_single_rte)
    full_data = full_data.remove_columns('sentence1')
    full_data = full_data.remove_columns('sentence2')
    # full_data = full_data.remove_columns('label')
    full_data = full_data.remove_columns('idx')

    # Sanity check
    print(full_data[0])
    if instruction_format:
        output_path = os.path.join(output_dir, "rte_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "rte.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["text_label"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "rte",
                "id": f"rte_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "rte",
                    "id": f"rte_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

    def format_single_gpt3nli(example):
        if example['label'] == 1:
            label = 'neutral'
        elif example['label'] == 0:
            label = 'entailment'
        else:
            label = 'contradiction'
        if instruction_format:
            example['input'] = f"Consider the following texts: Text 1: {example['text_a']} Text 2: {example['text_b']} The relation is "
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['text_a'] + '\n ### Input_2: ' + example['text_b'] + '\n ### Inference: '
            example['text'] = '### Input_1: ' + example['text_a'] + '\n ### Input_2: ' + example['text_b'] + '\n ### Inference: ' + label
        example['metadata'] = 'gpt3nli'
        example['text_label'] = label
        return example

    # Train set of RTE because hidden test sets
    raw_datasets = load_dataset('pietrolesci/gpt3_nli', split='train')
    
    # Arrange the data format
    full_data = raw_datasets.map(format_single_gpt3nli)
    full_data = full_data.remove_columns('text_a')
    full_data = full_data.remove_columns('text_b')
    # full_data = full_data.remove_columns('label')
    full_data = full_data.remove_columns('guid')

    # Sanity check
    print(full_data[0])
    if instruction_format:
        output_path = os.path.join(output_dir, "gpt3nli_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "gpt3nli.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["text_label"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "gpt3nli",
                "id": f"gpt3nli_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "gpt3nli",
                    "id": f"gpt3nli_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_paws(output_dir, use_prompt_format=True, get_test=False, instruction_format='instruct'):
    """
    data source: https://huggingface.co/datasets/paws
    Should only be ran once
    """
    def format_single_paws(example):
        label = 'yes' if example['label'] == 1 else 'no'
        if instruction_format == 'instruct':
            example['input'] = f"Let’s compare the two sentences: Sentence_1: {example['sentence1']} Sentence_2: {example['sentence2']} Are they paraphrasing?: "
            example['text'] = example['input'] + label
        elif instruction_format == 'inputoutput':
            example['input'] = f"{example['sentence1']} {example['sentence2']}"
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['sentence1'] + '\n ### Input_2: ' + example['sentence2'] + '\n ### Paraphrase Classification: '
            example['text'] = '### Input_1: ' + example['sentence1'] + '\n ### Input_2: ' + example['sentence2'] + '\n ### Paraphrase Classification: ' + label
        example['text_label'] = label
        example['metadata'] = 'paws'
        return example

    raw_datasets = load_dataset("paws", "labeled_final", ignore_verifications=True)
    # Shuffle and sample the training set
    raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42).select(range(6000))
    # Arrange the data format
    full_data = raw_datasets.map(format_single_paws)
    full_data = full_data.remove_columns('sentence1')
    full_data = full_data.remove_columns('sentence2')
    full_data = full_data.remove_columns('label')

    if get_test:
        if instruction_format == 'instruct' or instruction_format == 'inputoutput':
            output_path = os.path.join(output_dir, "evaluation", f"paws_{instruction_format}.jsonl")
        else:
            output_path = os.path.join(output_dir, "evaluation", "paws.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(full_data['test']):
                prompt = example["input"]
                completion = example["text_label"]
                if use_prompt_format:
                    fout.write(json.dumps({
                    "dataset": "paws",
                    "id": f"paws_{idx}",
                    "prompt": prompt,
                    "completion": completion
                }) + "\n")
                else:
                    fout.write(json.dumps({
                        "dataset": "paws",
                        "id": f"paws_{idx}",
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion},
                        ]
                    }) + "\n")
    else:
        output_path = os.path.join(output_dir, "paws.jsonl")
        with open(output_path, "w") as fout:
            for idx, example in enumerate(full_data['train']):
                prompt = example["input"]
                completion = example["text_label"]
                if use_prompt_format:
                    fout.write(json.dumps({
                    "dataset": "paws",
                    "id": f"paws_{idx}",
                    "prompt": prompt,
                    "completion": completion
                }) + "\n")
                else:
                    fout.write(json.dumps({
                        "dataset": "paws",
                        "id": f"paws_{idx}",
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": completion},
                        ]
                    }) + "\n")

def process_paraphrase_detection_test_sets(output_dir, use_prompt_format=True, instruction_format=False):
    """
    data source: qqp (https://huggingface.co/datasets/SetFit/qqp), STS-B (https://huggingface.co/datasets/tweet_qa)
    """
    def format_single_qqp(example):
        label = 'yes' if example['label'] == 1 else 'no'
        if instruction_format:
            example['input'] = f"Let’s compare the two sentences: Sentence_1: {example['text1']} Sentence_2: {example['text2']} Are they paraphrasing?: "
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['text1'] + '\n ### Input_2: ' + example['text2'] + '\n ### Paraphrase Classification: '
            example['text'] = '### Input_1: ' + example['text1'] + '\n ### Input_2: ' + example['text2'] + '\n ### Paraphrase Classification: ' + label
        example['text_label'] = label
        example['metadata'] = 'qqp'
        return example

    raw_datasets = load_dataset('SetFit/qqp', split='test')
    # Arrange the data format
    full_data = raw_datasets.map(format_single_qqp)
    full_data = full_data.remove_columns('label_text')
    full_data = full_data.remove_columns('idx')
    full_data = full_data.remove_columns('text1')
    full_data = full_data.remove_columns('text2')
    # full_data = full_data.remove_columns('label')
    
    if instruction_format:
        output_path = os.path.join(output_dir, "qqp_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "qqp.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["text_label"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "qqp",
                "id": f"qqp_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "qqp",
                    "id": f"qqp_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

    def format_single_stsb(example):
        # STS-b uses a 1 to 5 scale points
        label = 'yes' if example['label'] > 3.5 else 'no'
        if instruction_format:
            example['input'] = f"Let’s compare the two sentences: Sentence_1: {example['text1']} Sentence_2: {example['text2']} Are they paraphrasing?: "
            example['text'] = example['input'] + label
        else:
            example['input'] = '### Input_1: ' + example['text1'] + '\n ### Input_2: ' + example['text2'] + '\n ### Paraphrase Classification: '
            example['text'] = '### Input_1: ' + example['text1'] + '\n ### Input_2: ' + example['text2'] + '\n ### Paraphrase Classification: ' + label
        example['metadata'] = 'stsb'
        example['text_label'] = label
        return example
    raw_datasets = load_dataset("SetFit/stsb", split='test')
    # Arrange the data format
    # Using the same processing function w/ QQP because the format is the same
    full_data = raw_datasets.map(format_single_stsb)
    full_data = full_data.remove_columns('label_text')
    full_data = full_data.remove_columns('idx')
    full_data = full_data.remove_columns('text1')
    full_data = full_data.remove_columns('text2')
    # full_data = full_data.remove_columns('label')

    if instruction_format:
        output_path = os.path.join(output_dir, "stsb_instruct.jsonl")
    else:
        output_path = os.path.join(output_dir, "stsb.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["text_label"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "stsb",
                "id": f"stsb_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "stsb",
                    "id": f"stsb_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

def process_tulu_v2(output_dir, percentage=0.1):
    """
    data source: https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture
    Should only be ran once
    """
    # Can be solved by streaming, or try downgrading dataset?
    raw_datasets = load_dataset('allenai/tulu-v2-sft-mixture')
    # Shuffle and sample the training set
    target_sample_length = int(len(raw_datasets['train']) * percentage)
    print(f"Sampling the data to {target_sample_length} samples")
    raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42).select(range(target_sample_length))

    write_path = output_dir
    output_path = os.path.join(write_path, f"tulu_{str(percentage)}_data.jsonl")

    with open(output_path, "w") as fout:
        for idx, example in enumerate(raw_datasets['train']):
            fout.write(json.dumps({
                    "dataset": example['dataset'],
                    "id": f"tulu_{example['id']}",
                    "messages": example['messages']
                }) + "\n")

def process_llmbar(output_dir, split, use_prompt_format=False):
    """
    data source: https://github.com/princeton-nlp/LLMBar/tree/main
    """
    base_path = os.path.join(os.environ['data_dir'], 'eval', 'llmbar', 'LLMBar')

    if '_' not in split:
        raw_datasets = load_dataset("json", data_files=os.path.join(base_path, split, "dataset.json"))
    else:
        raw_datasets = load_dataset("json", data_files=os.path.join(base_path, split.split('_')[0], split.split('_')[1], "dataset.json"))

    def format_single_llmbar(example):
        orig_input = example['input']
        example['input'] = '### Instruction: ' + orig_input +'### Output_1: ' + example['output_1'] + '\n ### Output_2: ' + example['output_2'] + '\n ### Instruction Following: '
        example['text'] =  '### Instruction: ' + orig_input +'### Output_1: ' + example['output_1'] + '\n ### Output_2: ' + example['output_2'] + '\n ### Instruction Following: ' + str(example['label'])
        example['metadata'] = 'llmbar'
        return example
    # Shuffle and sample the training set
    raw_datasets = raw_datasets['train'].shuffle(seed=42)
    raw_datasets = raw_datasets.map(format_single_llmbar)
    raw_datasets = raw_datasets.remove_columns('output_1')
    raw_datasets = raw_datasets.remove_columns('output_2')


    write_path = output_dir
    output_path = os.path.join(write_path, f"llmbar_{split}.jsonl")

    with open(output_path, "w") as fout:
        for idx, example in enumerate(raw_datasets):
            prompt = example["input"]
            completion = str(example["label"])
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": f"llmbar_{split}",
                "id": f"llmbar_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": f"llmbar_{split}",
                    "id": f"llmbar_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")

### Adding this for sanity check between Tango Code v.s. This code
def process_sciq_test_sets(output_dir, use_prompt_format=True):
    """
    data source: sciq (https://huggingface.co/datasets/sciq)
    """
    def format_single_sciq(example):
        example['input'] = '### Input: ' + example['support'] + '\n ### Question: ' +  example['question'] + '\n ### Answer: Answer A: ' + example['correct_answer'] + ' Answer B: ' + example['distractor1'] + ' Answer C: ' + example['distractor2'] + ' Answer D: ' + example['distractor3'] + '\n ### Correct Answer: '
        example['text'] = '### Input: ' + example['support'] + '\n ### Question: ' +  example['question'] + '\n ### Answer: Answer A: ' + example['correct_answer'] + ' Answer B: ' + example['distractor1'] + ' Answer C: ' + example['distractor2'] + ' Answer D: ' + example['distractor3'] + '\n ### Correct Answer: ' + example['correct_answer']
        example['metadata'] = 'sciq'
        return example

    raw_datasets = load_dataset('sciq', split='test')
    # Arrange the data format
    full_data = raw_datasets.map(format_single_sciq)
    full_data = full_data.remove_columns('support')
    # full_data = full_data.remove_columns('correct_answer')
    full_data = full_data.remove_columns('distractor1')
    full_data = full_data.remove_columns('distractor2')
    full_data = full_data.remove_columns('distractor3')
    
    # Sanity check
    print(full_data[0])
    output_path = os.path.join(output_dir, "sciq_multichoice.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(full_data):
            prompt = example["input"]
            completion = example["correct_answer"]
            if use_prompt_format:
                fout.write(json.dumps({
                "dataset": "sciq",
                "id": f"sciq_{idx}",
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            else:
                fout.write(json.dumps({
                    "dataset": "sciq",
                    "id": f"sciq_{idx}",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]
                }) + "\n")


if __name__ == "__main__":
    process_summarization_test_set(output_dir=os.environ['data_dir'] + '/evaluation', instruction_format='inputoutput')
    process_socialiqa_test(output_dir=os.environ['data_dir'] + '/evaluation', instruction_format='inputoutput')
    process_question_generation_test_sets(output_dir=os.environ['data_dir'] + '/evaluation', instruction_format=True)
    process_question_generation_test_sets(output_dir=os.environ['data_dir'] + '/evaluation', instruction_format='instruct')
    process_mnli(output_dir=os.environ['data_dir'], use_prompt_format=True, get_test=True, mismatched=False, instruction_format='inputoutput')
    process_mnli(output_dir=os.environ['data_dir'], use_prompt_format=True, get_test=True, mismatched=True, instruction_format='inputoutput')
    process_mnli(output_dir=os.environ['data_dir'], use_prompt_format=True, get_test=True, mismatched=False, instruction_format=True)
    process_socialiqa(output_dir='/workspace/open-instruct/data', use_prompt_format=True)
    process_nli_test(output_dir=os.environ['data_dir'] + '/evaluation', use_prompt_format=True, instruction_format=True)
    process_paws(output_dir=os.environ['data_dir'], use_prompt_format=True, get_test=False)
    process_paws(output_dir=os.environ['data_dir'], use_prompt_format=True, get_test=True, instruction_format='inputoutput')
    process_paraphrase_detection_test_sets(output_dir=os.environ['data_dir'] + '/evaluation', use_prompt_format=True, instruction_format=True)
    process_tulu_v2(output_dir=os.environ['data_dir'], percentage=0.05)
    process_llmbar(output_dir=os.environ['data_dir'] + '/evaluation', split='Natural')
    process_llmbar(output_dir=os.environ['data_dir'] + '/evaluation', split='Adversarial_Manual')
    process_llmbar(output_dir=os.environ['data_dir'] + '/evaluation', split='Adversarial_Neighbor')
    process_llmbar(output_dir=os.environ['data_dir'] + '/evaluation', split='Adversarial_GPTOut')
    process_llmbar(output_dir=os.environ['data_dir'] + '/evaluation', split='Adversarial_GPTInst')

    process_mnli(output_dir=os.environ['data_dir'], use_prompt_format=True, get_test=True, mismatched=True)
    process_sciq_test_sets(output_dir=os.environ['data_dir'] + '/evaluation', use_prompt_format=True)