#!/bin/bash

keys=../data/tsp_seq.tsv
locks=../data/sero_k_seq.tsv

#python embed_proteins.py --input_file tsp_seq.tsv --output_file keys_embeddings.npz
python ../src/embed_proteins.py --input_file $locks --output_file ../output/locks_embeddings.npz

python ../src/visualize_embeddings.py --embeddings ../output/locks_embeddings.npz ../output/keys_embeddings.npz --labels locks keys --output ../output/k_umap_plot.png

python extract_clusters.py --coords_file ../output/k_umap_plot.coords.csv --output_dir ../output/clusters_dbscan --method dbscan --eps 0.5
python extract_clusters.py --coords_file ../output/k_umap_plot.coords.csv --output_dir ../output/clusters_kmeans --method kmeans --n_clusters 5


