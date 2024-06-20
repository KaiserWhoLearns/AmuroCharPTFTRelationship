
# To run this, download OLMO checkpoints using open_instruct/download_olmo_ckpts.py
# Run the generated outputs/wget_ckpts.sh to wget the checkpoints
# Then, execute this script to make OLMO ckpts into HF format
for ckpt in 342000 424000 505000 592000 738000; do
    python open_instruct/olmo_hf/convert_olmo_to_hf.py --checkpoint-dir ${base_dir}/output/olmo1b_ckpts/checkpoint-${ckpt}
done