# Amuro and Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of Large Language Models

This is the repository to reproduce the experiments in Amuro and Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of Large Language Models.

Much of the code is adapted from [AI2 Open Instruct](https://github.com/allenai/open-instruct/tree/main) and [OLMo Eval](https://github.com/allenai/OLMo-Eval) repository.
Thanks to the authors of these repositories 🤗

## 📋 Preparation

### 🐍 Create environment
```
conda env create --file=environments_eval.yml
conda activate eval-pipeline
```
Before execution, please set the environment variable `base_dir` and `data_dir` to point to the directory of this folder and directory of data correspondingly.

### 🗂️ Retrieve and format datasets
The datasets used for supervised fine-tuning can be retrieved and format with the functions in `open_instruct/format_genearlization.py`. By default, this script will format all the datasets, including different task formats.
The training set will be saved to `$data_dir` and the evaluation set will be saved to `$data_dir/evaluation` as json files.

The datasets used for instruction tuning can be loaded in the same way as [OLMo Eval](https://github.com/allenai/OLMo-Eval), which uses AI2 CatWalk.

### 🎯 Gather Intermediate Pre-training Checkpoints
Intermediate checkpoints that are used for experiments can be downloaded using the script `open_instruct/download_olmo_ckpts.py`.

Since this repository by default uses Huggingface training and evaluation pipeline, the OLMo checkpoints need to be converted to HF format using `open_instruct/olmo_hf/convert_olmo_to_tf.py`.
Warning: Training the models in wrong format may lead to [huge performance degradation](https://github.com/allenai/OLMo-Eval/issues/31)!

OLMo team also recently made later version of OLMo checkpoints available through [Huggingface Hub](https://huggingface.co/allenai/OLMo-1.7-7B-hf/tree/main).

## 🏋️‍♂️ Fine-Tuning

All the fine-tuned checkpoints can be found through Huggingface Hub. This section presents a way to reproduce the fine-tuning we conducted.

After preparing all the recipes, we can start ~~cooking~~ fine-tuning the checkpoints.

For both supervised fine-tuning and instruction tuning, the script `scripts/slurm_scripts/find_tune_olmo.sh` can be used to submit a slurm job to train the corresponding checkpoint with specified dataset and hyperparameters.

The trained models will by default be saved to `output/`, with each folder name indicating the checkpoint, training data, number of epochs, and learning rate.

## 📈 Evaluation
### 🔬 Supervised Fine-tuning
Before running the evaluation, `generate_pred.sh` should be executed to produce the corresponding generation. By default, the predictions will be written to `output/predictions/` as json files.

There are multiple scripts of evaluating the model. To only evaluate a single model variant, use `eval_pred.py`

To evaluate all available models predictions in `output/predictions/`, one can run `analysis/produce_tables.py`.

### 💡 Instruction Tuning
The instruction tuned model evaluation followed the official OLMo evaluation pipeline.
Metrics will be automatically written to `{data_name}_${model_name}.json`.

```
python -m olmo_eval.run_lm_eval --model-path ${model_path} \
    --task $data_name --split validation \
    --num-shots $num_shots \
    --metrics-file ${base_dir}/eval_results/eval_pipeline/${data_name}_${model_name}.json
```

## 🧐 Analysis

To reproduce the figures in the paper, use the notebook `analysis/gen_paper_figs.ipynb`.
By default, it will generate the figures with our released performance numbers (`results/analysis/all_perf_table.csv`). The file should be replaced if custom model setup is preferred.

## © License
Much of the code is adapted from [AI2 Open Instruct](https://github.com/allenai/open-instruct/tree/main) and [OLMo Eval](https://github.com/allenai/OLMo-Eval) repository, which are under Apcache 2.0 license.

The remaining of the code (`analysis/`, `scripts/slurm_scripts/`, `eval/predict.py`, `format_generalization,py`, `open_instruct/download_olmo_ckpts.py`) are under MIT license.

## 🫶 Citation ´･ᴗ･`
```
@article{sun2024amuro,
  title={Amuro \& char: Analyzing the relationship between pre-training and fine-tuning of large language models},
  author={Sun, Kaiser and Dredze, Mark},
  journal={arXiv preprint arXiv:2408.06663},
  year={2024}
}
```
