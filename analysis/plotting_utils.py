import os
import pdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# The file to generate plots

LRS = {
    'xsum': '2e-7',
    'socialiqa': '2e-6',
    'mnli': '2e-6',
    'paws': '2e-6',
    'tulu': '2e-6',
}

def ckpt_vs_perf_plot(eval_dataset, base_dataset, add_multi_shots=0, show_both=False):
    # Generate the figure the produce checkpoint v.s. performance plot
    if base_dataset == 'tulu' and 'llmbar' not in eval_dataset:
        all_perf = pd.read_csv(os.path.join(os.environ['base_dir'], 'results', 'analysis', 'it_perf_table.csv'))
    else:
        all_perf = pd.read_csv(os.path.join(os.environ['base_dir'], 'results', 'analysis', 'all_perf_table.csv'))

    ft_perfs = []
    orig_perfs = []
    ft_stds = []
    orig_stds = []
    multi_shots_ft = []
    multi_shots_orig = []
    multi_shots_ft_stds = []
    multi_shots_orig_stds = []
    checkpoints = ['1000', '18000', '342000', '424000', '505000', '592000', '738000', 'main']
    epoch = '5' if base_dataset == 'tulu' else '3'

    # Load prediction of the corresponding eval dataset, for each checkpoint
    # Both original and fine-tuned
    for ckpt in checkpoints:
        if ckpt != 'main':
            model_id = 'olmo1b_hf_ckpt' + ckpt + '_' + base_dataset + '_' + epoch +'epoch_' + LRS[base_dataset]
            if base_dataset == 'tulu' and 'llmbar' not in eval_dataset:
                # Instruction tuning base results
                orig_model_id = 'checkpoint-' + ckpt
            else:
                orig_model_id = 'olmo1b_checkpoint-' + ckpt + '_original'
        else:
            model_id = 'olmo1b_hf_main_' + base_dataset + '_' + epoch + 'epoch_' + LRS[base_dataset]
            orig_model_id = 'olmo1b_original_hf'
        ft_perf = all_perf.loc[(all_perf['model_id'] == model_id) & (all_perf['eval dataset'] == eval_dataset)]
        # Load the original model
        orig_perf = all_perf.loc[(all_perf['model_id'] == orig_model_id) & (all_perf['eval dataset'] == eval_dataset)]
        if len(ft_perf) == 1:
            ft_perfs.append(ft_perf['Performance'].item())
            if 'std' in ft_perf:
                ft_stds.append(ft_perf['std'].item())
        else:
            ft_perfs.append(None)
        if len(orig_perf) == 1:
            orig_perfs.append(orig_perf['Performance'].item())
            if 'std' in orig_perf:
                orig_stds.append(orig_perf['std'].item())
        else:
            orig_perfs.append(None)
        
        if add_multi_shots > 0:
            if ckpt != 'main':
                model_id = 'olmo1b_hf_ckpt' + ckpt + '_' + base_dataset + '_' + epoch +'epoch_' + LRS[base_dataset] + f'_{str(add_multi_shots)}shots'
                if base_dataset == 'tulu' and 'llmbar' not in eval_dataset:
                    # Instruction tuning base results
                    orig_model_id = 'checkpoint-' + ckpt
                else:
                    orig_model_id = 'olmo1b_checkpoint-' + ckpt + f'_original_hf_{str(add_multi_shots)}shots'
            else:
                model_id = f'olmo1b_hf_main_{base_dataset}_{epoch}epoch_{LRS[base_dataset]}_{str(add_multi_shots)}shots'
                orig_model_id = f'olmo1b_original_hf_{str(add_multi_shots)}shots'
            ft_perf = all_perf.loc[(all_perf['model_id'] == model_id) & (all_perf['eval dataset'] == eval_dataset)]
            # Load the original model
            orig_perf = all_perf.loc[(all_perf['model_id'] == orig_model_id) & (all_perf['eval dataset'] == eval_dataset)]
            
            if len(ft_perf) == 1:
                multi_shots_ft.append(ft_perf['Performance'].item())
                multi_shots_ft_stds.append(ft_perf['std'].item())
            else:
                multi_shots_ft.append(None)
                multi_shots_ft_stds.append(None)
            if len(orig_perf) == 1:
                multi_shots_orig.append(orig_perf['Performance'].item())
                multi_shots_orig_stds.append(orig_perf['std'].item())
            else:
                multi_shots_orig.append(None)
                multi_shots_orig_stds.append(None)
            
    if add_multi_shots == 0:
        data_to_plot = pd.DataFrame({
            'Performance': ft_perfs + orig_perfs,
            'Fine-tuned': ['FT' for _ in range(len(ft_perfs))] + ['Orig' for _ in range(len(orig_perfs))],
            'ckpt_idx': [i for i in range(len(checkpoints))] + [i for i in range(len(checkpoints))]
            })
    if add_multi_shots > 0:
        low1, high1, low2, high2, fill_x = [], [], [], [], []
        for i in range(len(multi_shots_ft)):
            if multi_shots_ft[i] is not None and multi_shots_orig[i] is not None:
                low1.append(multi_shots_ft[i] - multi_shots_ft_stds[i])
                high1.append(multi_shots_ft[i] + multi_shots_ft_stds[i])
                low2.append(multi_shots_orig[i] - multi_shots_orig_stds[i])
                high2.append(multi_shots_orig[i] + multi_shots_orig_stds[i])
                fill_x.append(i)
        data_to_plot = pd.DataFrame({
        'Performance': multi_shots_ft + multi_shots_orig,
        'Fine-tuned': [str(add_multi_shots) + 'shotsFT' for _ in range(len(ft_perfs))] + [str(add_multi_shots) + 'shotsOrig' for _ in range(len(orig_perfs))],
        'ckpt_idx': [i for i in range(len(checkpoints))] + [i for i in range(len(checkpoints))]
        })
    if show_both and add_multi_shots > 0:
        data_to_plot = pd.DataFrame({
        'Performance': ft_perfs + orig_perfs + multi_shots_ft + multi_shots_orig,
        'Fine-tuned': ['FT' for _ in range(len(ft_perfs))] + ['Orig' for _ in range(len(orig_perfs))] + [str(add_multi_shots) + 'shotsFT' for _ in range(len(ft_perfs))] + [str(add_multi_shots) + 'shotsOrig' for _ in range(len(orig_perfs))],
        'ckpt_idx': [i for i in range(len(checkpoints))] + [i for i in range(len(checkpoints))] + [i for i in range(len(checkpoints))] + [i for i in range(len(checkpoints))]
        })
    # Create the plot
    # Uncomment if fitting a regression line
    # dist_plot = sns.lmplot(data=data_to_plot, x="ckpt_idx", y="Performance", hue="Fine-tuned", ci=95, robust=True, legend_out=False)
    dist_plot = sns.lineplot(data=data_to_plot, x="ckpt_idx", y="Performance", marker='o', style="Fine-tuned", hue="Fine-tuned", legend="auto")
    # Uncomment to plot confidence intervals
    if len(multi_shots_ft_stds) > 0:
        plt.fill_between(fill_x, low1, high1, alpha=0.4)
        plt.fill_between(fill_x, low2, high2, alpha=0.4)
    # dist_plot = sns.lmplot(data=data_to_plot, x="ckpt_idx", y="Performance", hue="Fine-tuned", ci=95, legend_out=False)
    dist_plot.set(xticks=[i for i in range(len(checkpoints))], xlim=[-0.2, len(checkpoints)-0.8], ylim=[0.0, 1.0], xlabel="Checkpoint Index", title=f"Eval:{eval_dataset}, Train:{base_dataset}")
    dist_plot.set_xticklabels(checkpoints, rotation=30)
    # Uncomment if using lmplot
    # dist_plot.savefig(os.path.join(os.environ['base_dir'], "results", "analysis", f"eval{eval_dataset}-train{base_dataset}.png"))
    if add_multi_shots > 0:
        plt.savefig(os.path.join(os.environ['base_dir'], "results", "analysis", f"eval{eval_dataset}-train{base_dataset}_{str(add_multi_shots)}shots.png"))
    else:
        plt.savefig(os.path.join(os.environ['base_dir'], "results", "analysis", f"eval{eval_dataset}-train{base_dataset}.png"))
    plt.clf()

def it_ckpt_vs_perf_plot(eval_dataset):
    # Get the performance table
    all_perf = pd.read_csv(os.path.join(os.environ['base_dir'], 'results', 'analysis', 'official_eval_table.csv'))
    # Gather
    ft_perfs = []
    orig_perfs = []
    checkpoints = ['1000', '18000', '342000', '424000', '505000', '592000', '738000', 'main']
    for ckpt in checkpoints:
        if ckpt != 'main':
            orig_model_id = 'checkpoint-' + ckpt
            ft_model_id = 'olmo1b_hf_ckpt' + ckpt + '_tulu_5epoch_2e-6'
        else:
            orig_model_id = 'olmo1b_original_hf'
            ft_model_id = 'olmo1b_hf_main_tulu_5epoch_2e-6'
        orig_perf = all_perf.loc[(all_perf['model_id'] == orig_model_id) & (all_perf['eval dataset'] == eval_dataset)]
        ft_perf = all_perf.loc[(all_perf['model_id'] == ft_model_id) & (all_perf['eval dataset'] == eval_dataset)]
        if len(orig_perf) == 1:
            orig_perfs.append(orig_perf['Performance'].item())
        else:
            orig_perfs.append(None)
        if len(ft_perf) == 1:
            ft_perfs.append(ft_perf['Performance'].item())
        else:
            ft_perfs.append(None)
    data_to_plot = pd.DataFrame({
        'Performance': orig_perfs + ft_perfs,
        'Variant': ['BASE' for _ in range(len(orig_perfs))] + ['Instruct' for _ in range(len(orig_perfs))],
        'ckpt_idx': [i for i in range(len(checkpoints))] + [i for i in range(len(checkpoints))]
        })
    dist_plot = sns.lineplot(data=data_to_plot, x="ckpt_idx", y="Performance", marker='o', style="Variant", hue="Variant", legend="auto")
    dist_plot.set(xticks=[i for i in range(len(checkpoints))], xlim=[-0.2, len(checkpoints)-0.8], ylim=[0.0, 1.0], xlabel="Checkpoint Index", title=f"Eval:{eval_dataset}")
    dist_plot.set_xticklabels(checkpoints, rotation=30)
    # pdb.set_trace()
    plt.savefig(os.path.join(os.environ['base_dir'], "results", "analysis", f"it_eval{eval_dataset}.png"))
    plt.clf()

if __name__ == '__main__':
    sns.set_theme()
    font = {'family' : 'serif',
            # 'weight' : 'bold',
            'size'   : 14}
    mpl.rcParams['figure.dpi'] = 600
    mpl.rc('font', **font)
    mpl.rc('xtick', labelsize=12) 
    plt.rcParams["font.family"] = "Nimbus Roman"
    mpl.rc('ytick', labelsize=12)
    ckpt_vs_perf_plot('mnli_matched', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('mnli_mismatched', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('rte', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('gpt3nli', 'mnli', add_multi_shots=4)

    ckpt_vs_perf_plot('paws', 'paws', add_multi_shots=4)
    ckpt_vs_perf_plot('stsb', 'paws', add_multi_shots=4)
    ckpt_vs_perf_plot('qqp', 'paws', add_multi_shots=4)

    ckpt_vs_perf_plot('socialiqa', 'socialiqa', add_multi_shots=4)
    ckpt_vs_perf_plot('tweetqa', 'socialiqa', add_multi_shots=4)
    ckpt_vs_perf_plot('sciq', 'socialiqa', add_multi_shots=4)
    
    ckpt_vs_perf_plot('xsum', 'xsum', add_multi_shots=2)
    ckpt_vs_perf_plot('xsum', 'xsum', add_multi_shots=4)
    ckpt_vs_perf_plot('xlsum', 'xsum', add_multi_shots=4)
    ckpt_vs_perf_plot('cnn', 'xsum', add_multi_shots=4)

    # Cross Task Generalization
    ckpt_vs_perf_plot('mnli_matched', 'socialiqa', add_multi_shots=4)
    ckpt_vs_perf_plot('mnli_matched', 'paws', add_multi_shots=4)
    ckpt_vs_perf_plot('socialiqa', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('socialiqa', 'paws', add_multi_shots=4)
    ckpt_vs_perf_plot('paws', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('paws', 'socialiqa', add_multi_shots=4)

    # Instruction format
    ckpt_vs_perf_plot('mnli_matched_instruct', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('rte_instruct', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('gpt3nli_instruct', 'mnli', add_multi_shots=4)
    ckpt_vs_perf_plot('xsum_instruct', 'xsum', add_multi_shots=4)
    ckpt_vs_perf_plot('xlsum_instruct', 'xsum', add_multi_shots=4)
    ckpt_vs_perf_plot('paws_instruct', 'paws', add_multi_shots=4)
    ckpt_vs_perf_plot('stsb_instruct', 'paws', add_multi_shots=4)
    ckpt_vs_perf_plot('socialiqa_instruct', 'socialiqa', add_multi_shots=4)
    ckpt_vs_perf_plot('sciq_instruct', 'socialiqa', add_multi_shots=4)
    

    # Instruction Tuning
    # ckpt_vs_perf_plot('llmbar_Natural', 'tulu', add_multi_shots=4)
    # ckpt_vs_perf_plot('mmlu', 'tulu', add_multi_shots=0)
    # ckpt_vs_perf_plot('toxigen', 'tulu', add_multi_shots=0)
    # ckpt_vs_perf_plot('truthfulqa', 'tulu', add_multi_shots=0)
    # ckpt_vs_perf_plot('gsm', 'tulu', add_multi_shots=0)

    # # Instruction Tuning
    # it_ckpt_vs_perf_plot(eval_dataset='arc_easy')
    # it_ckpt_vs_perf_plot(eval_dataset='arc_challenge')
    # it_ckpt_vs_perf_plot(eval_dataset='sciq')
    # it_ckpt_vs_perf_plot(eval_dataset='boolq')
    # it_ckpt_vs_perf_plot(eval_dataset='hellaswag')
    # it_ckpt_vs_perf_plot(eval_dataset='openbookqa')