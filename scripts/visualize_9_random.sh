#!/bin/bash

keys=tsp_seq.tsv
locks=sero_seq.tsv


python filter_by_genomes.py --locks_file $locks --keys_file $keys --output_locks filtered_locks.tsv --output_keys filtered_keys.tsv --n_genomes 9

python embed_proteins.py --input_file filtered_locks.tsv --output_file filtered_locks_embed.npz

python embed_proteins.py --input_file filtered_keys.tsv --output_file filtered_keys_embed.npz

python visualize_embeddings.py --embeddings filtered_locks_embed.npz filtered_keys_embed.npz --labels locks keys --output umap_plot.png --grid_subplots

