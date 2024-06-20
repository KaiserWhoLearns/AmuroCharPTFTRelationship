import argparse
import pdb
import os
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Grab the intermediate pre-training checkpoints from OLMO")
    parser.add_argument(
        "--ckpt_amount",
        type=int,
        default=0,
        help="The amount of checkpoint to grab.",
    )
    parser.add_argument(
        "--download_in_parallel",
        action="store_true",
        help="Whether to download in parallel.",
    )
    parser.add_argument(
        "--get_wget_list",
        action="store_true",
        help="Whether to download directory or get a list of wget command",
    )
    parser.add_argument(
        "--ckpt_csv",
        type=str,
        default='ds_configs/OLMo-1B.csv',
        help="The dir that contain the checkpoint links.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output dir to store the checkpoints.",
    )
    args = parser.parse_args()
    return args

def download_file(idx, urls, write_dir):
    url = urls[idx]
    response = requests.get(url)
    if "content-disposition" in response.headers:
        content_disposition = response.headers["content-disposition"]
        filename = content_disposition.split("filename=")[1]
    else:
        filename = url.split("/")[-1]
    print(f"Downloading {filename}")
    with open(os.path.join(write_dir, filename), mode="wb") as file:
        file.write(response.content)
    print(f"Downloaded file {filename}")

def main():
    args = parse_args()
    # python open_instruct/download_olmo_ckpts.py \
    # --ckpt_amount 5 \
    # --ckpt_csv ${base_dir}/ds_configs/OLMo-1B.csv \
    # --download_in_parallel \
    # --get_wget_list \
    # --output_dir ${base_dir}/output/olmo1b_ckpts
    # Load the checkpoint cloud locations
    csv_path = args.ckpt_csv
    # Step, Checkpoint Directory
    cloud_locs = pd.read_csv(csv_path)
    all_commands = []

    # Get the target to be downloaded
    ckpts_steps_to_download = []
    indices = []
    if args.ckpt_amount != 0:
        ckpts_steps_to_download = []
        indices = []
        # Count the checkpoints to be downloaded
        intermediate_steps = int(len(cloud_locs) / args.ckpt_amount)
        for idx in range(intermediate_steps-1, len(cloud_locs), intermediate_steps):
            ckpts_steps_to_download.append(cloud_locs['Step'][idx])
            indices.append(idx)
        assert len(ckpts_steps_to_download) == args.ckpt_amount
        
    # download the checkpoints
    for idx, steps in tqdm(enumerate(ckpts_steps_to_download)):
        template_url = (
            cloud_locs['Checkpoint Directory'][indices[idx]] + "{resource}"
        )
        # config.yaml: the config at that training step.
        # model.pt, optim.pt, train.pt: model, optimizer and training state at that training step.
        urls = [
            template_url.format(resource="config.yaml"),
            template_url.format(resource="model.pt"),
            template_url.format(resource="optim.pt"),
            template_url.format(resource="train.pt"),
        ]
        write_dir = os.path.join(args.output_dir, f'checkpoint-{steps}')

        if args.get_wget_list:
            commands = [f"wget -P {write_dir} {url} \n" for url in urls]
            all_commands += commands
        else:
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)
            
            print(urls, write_dir)
            
            if args.download_in_parallel:
                # Download files in parallel
                with ProcessPoolExecutor() as executor:
                    worker = partial(download_file, urls=urls, write_dir=write_dir)
                    mmp = executor.map(worker, range(0, len(urls)))
                    for r in mmp:
                        # Output the exception if there is one
                        print(r)
            else:
                for idx in range(0, len(urls)):
                    download_file(idx, urls=urls, write_dir=write_dir)
                        
            print("Finish downloading.")
    if args.get_wget_list:
        print(all_commands)
        file = open('output/wget_ckpts.sh','w')
        file.writelines(all_commands)
        file.close()

if __name__ == "__main__":
    main()