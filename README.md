# ESM-C Protein Embeddings

A comprehensive PyTorch pipeline for generating, analyzing, and visualizing protein embeddings using ESM-C models from EvolutionaryScale.

## Overview

This repository provides tools for:
- **Embedding Generation**: Convert protein sequences to embeddings using ESM-C models
- **Data Filtering**: Select specific subsets of sequences for analysis
- **Similarity Analysis**: Calculate pairwise similarities between protein sets
- **Visualization**: Create UMAP visualizations of embedding spaces
- **Clustering**: Extract and analyze clusters from embeddings

## Installation

```bash
# Clone the repository
git clone https://github.com/leannmlindsey/esmc_protein_embed.git
cd esmc_protein_embed

# Create conda environment 
conda create -n esmc python=3.10
conda activate esmc

# Install dependencies
pip install torch transformers
pip install numpy pandas scikit-learn
pip install tqdm
pip install esm
pip install matplotlib
pip install seaborn
pip install umap-learn
```

## Quick Start

### 1. Generate Embeddings

```bash
# Using default ESM-C 300M model
python embed_proteins.py --input_file example_proteins.tsv --output_file output/embeddings.npz

# Using ESM-C 600M model for better accuracy
python embed_proteins.py --input_file proteins.tsv --output_file output/embeddings.npz --model esmc_600m
```

### 2. Visualize Embeddings

```bash
# Visualize all embeddings with UMAP
python visualize_embeddings.py --embeddings output/embeddings.npz --labels proteins --output output/umap_plot.png

# Or use t-SNE for better separation
python visualize_embeddings.py --embeddings output/embeddings.npz --labels proteins --output output/tsne_plot.png --method tsne

# Visualize locks and keys separately with grid
python visualize_embeddings.py --embeddings output/locks.npz output/keys.npz --labels locks keys --output output/umap_plot.png --grid_subplots
```

### 3. Calculate Similarities

```bash
# Calculate similarities between lock and key sets
python calculate_similarity.py --locks_file output/locks.npz --keys_file output/keys.npz --output_file output/similarities.csv
```

## Core Components

### 1. Embedding Generation (`embed_proteins.py`)

Converts protein sequences to high-dimensional embeddings using ESM-C models.

**Features:**
- Support for multiple ESM-C models (300M, 600M)
- Batch processing for efficiency
- Multiple output formats (NPZ, PKL, NPY)
- Automatic GPU detection

**Usage:**
```bash
python embed_proteins.py --input_file <input.tsv> --output_file output/<embeddings.npz> [options]
```

**Options:**
- `--model`: Choose ESM-C model (default: esmc_300m)
  - `esmc_300m`: Faster, less memory
  - `esmc_600m`: More accurate, requires more resources
- `--batch_size`: Processing batch size (default: 8)
- `--format`: Output format - npz, pkl, or npy
- `--sequence_col`: Column name for sequences (default: 'sequence')
- `--id_col`: Column name for IDs (default: 'id')

### 2. Data Filtering Tools

#### Filter Single Pairs (`filter_single_pairs.py`)
Extracts only groups with exactly one lock and one key sequence.

```bash
python filter_single_pairs.py --locks_file locks.tsv --keys_file keys.tsv \
    --output_locks output/filtered_locks.tsv --output_keys output/filtered_keys.tsv
```

#### Filter by Genomes (`filter_by_genomes.py`)
Selects all sequences from specific genomes.

```bash
# Select 50 random genomes
python filter_by_genomes.py --locks_file locks.tsv --keys_file keys.tsv \
    --output_locks output/subset_locks.tsv --output_keys output/subset_keys.tsv --n_genomes 50

# Select specific genomes
python filter_by_genomes.py --locks_file locks.tsv --keys_file keys.tsv \
    --output_locks output/subset_locks.tsv --output_keys output/subset_keys.tsv \
    --genome_list GCA_000016305.1 GCA_000163455.1
```

### 3. Similarity Calculation (`calculate_similarity.py`)

Calculates cosine similarities between lock and key embeddings within matching groups.

**Features:**
- Automatic group detection by sequence ID prefix
- Identifies top-scoring matches
- Outputs detailed statistics

```bash
python calculate_similarity.py --locks_file output/locks.npz --keys_file output/keys.npz \
    --output_file output/similarities.csv
```

### 4. Visualization (`visualize_embeddings.py`)

Creates UMAP or t-SNE visualizations of embedding spaces with multiple display options.

**Features:**
- Choice of UMAP or t-SNE dimensionality reduction
- Single or multiple embedding files
- Color coding by sequence type (locks/keys)
- Grid visualization for individual groups
- Customizable parameters for both methods

```bash
# Basic UMAP visualization
python visualize_embeddings.py --embeddings output/embeddings.npz --labels sequences \
    --output output/umap_plot.png

# Using t-SNE instead of UMAP
python visualize_embeddings.py --embeddings output/embeddings.npz --labels sequences \
    --output output/tsne_plot.png --method tsne

# Separate locks and keys with grid view
python visualize_embeddings.py --embeddings output/locks.npz output/keys.npz \
    --labels locks keys --output output/umap_plot.png --grid_subplots

# t-SNE with custom parameters
python visualize_embeddings.py --embeddings output/locks.npz --labels locks \
    --output output/tsne_locks.png --method tsne --perplexity 50 --n_iter 2000
```

**Options:**
- `--method`: Choose 'umap' or 'tsne' (default: umap)
- **UMAP parameters:**
  - `--n_neighbors`: Connectivity (default: 15)
  - `--min_dist`: Minimum distance (default: 0.1)
- **t-SNE parameters:**
  - `--perplexity`: Balance between local and global aspects (default: 30)
  - `--n_iter`: Number of iterations (default: 1000)
  - `--learning_rate`: Learning rate (default: auto)
- **Common options:**
  - `--metric`: Distance metric (default: cosine)
  - `--grid_subplots`: Create 3x3 grid of individual groups
  - `--group_by_prefix`: Color by sequence group prefix

### 5. Cluster Extraction (`extract_clusters.py`)

Identifies and extracts clusters from UMAP or t-SNE coordinates.

```bash
# Using DBSCAN
python extract_clusters.py --coords_file output/umap_plot.coords.csv \
    --output_dir output/clusters --method dbscan --eps 0.5

# Using K-means with 5 clusters
python extract_clusters.py --coords_file output/umap_plot.coords.csv \
    --output_dir output/clusters --method kmeans --n_clusters 5
```

## Input File Format

All tools expect tab-delimited files with at least two columns:

```
id	sequence
GCA_000016305.1_03570	MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
GCA_000016305.1_04233	KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE
```

## Output Formats

- **Embeddings**: NPZ (recommended), PKL, or NPY format
- **Similarities**: CSV with columns: group, lock_id, key_id, similarity, is_top_score
- **Visualizations**: PNG images with optional coordinate CSV files
- **Clusters**: Text files with sequence IDs and visualization plots

## GPU Support

The pipeline automatically detects and uses CUDA GPUs when available. For CPU-only processing, the code will automatically fall back to CPU mode.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- At least 16GB RAM (32GB for larger datasets)

## License

MIT License
