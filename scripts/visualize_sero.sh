#!/bin/bash

keys=tsp_seq.tsv
locks=sero_seq.tsv

#python embed_proteins.py --input_file tsp_seq.tsv --output_file keys_embeddings.npz
#python embed_proteins.py --input_file sero_seq.tsv --output_file locks_embeddings.npz

python visualize_embeddings.py --embeddings locks_embeddings.npz keys_embeddings.npz --labels locks keys --output umap_plot.png

python extract_clusters.py --coords_file umap_plot.coords.csv --output_dir clusters_dbscan --method dbscan --eps 0.5
python extract_clusters.py --coords_file umap_plot.coords.csv --output_dir clusters_kmeans --method kmeans --n_clusters 5


