# ESM-C Protein Embeddings

A PyTorch implementation for generating protein embeddings using the ESM-C (EvolutionaryScale/esmc-300m-2024-12) model.

## Features

- Load protein sequences from tab-delimited files
- Generate embeddings using ESM-C model
- Batch processing for efficient GPU utilization
- Multiple output formats (npz, pkl, npy)
- GPU support with automatic device detection

## Installation

```bash
# Clone the repository
git clone https://github.com/leannmlindsey/esmc_protein_embed.git
cd esmc_protein_embed

# Create conda environment 
conda create -n esmc python=3.10
conda activate esmc
pip install torch transformers
pip install numpy pandas scikit-learn
pip install tqdm
pip install esm
pip install matplotlib
pip install seaborn
pip install umap-learn


```

## Usage

### Basic Usage

```bash
python embed_proteins.py --input_file example_proteins.tsv --output_file embeddings.npz
```

### Command Line Arguments

- `--input_file`: Path to tab-delimited input file (required)
- `--output_file`: Path to save embeddings (required)
- `--sequence_col`: Column name for sequences (default: 'sequence')
- `--id_col`: Column name for sequence IDs (default: 'id')
- `--batch_size`: Batch size for processing (default: 8)
- `--format`: Output format - npz, pkl, or npy (default: 'npz')

### Input File Format

The input file should be tab-delimited with at least two columns:
- `id`: Unique identifier for each protein
- `sequence`: Amino acid sequence

Example:
```
id	sequence
protein1	MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG
protein2	KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE
```

### Output Formats

1. **NPZ**: Compressed NumPy archive with embeddings keyed by protein ID
2. **PKL**: Python pickle file containing a dictionary of embeddings
3. **NPY**: NumPy array of embeddings with separate ID file

## GPU Support

The code automatically detects and uses GPU if available. To force CPU usage:
```python
embedder = ESMCEmbedder(device='cpu')
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

## Similarity Calculation

The repository includes a script to calculate cosine similarities between two sets of embeddings (locks and keys), grouped by sequence ID prefix.

### Usage

```bash
python calculate_similarity.py --locks_file locks_embeddings.npz --keys_file keys_embeddings.npz --output_file similarities.csv
```

### Features

- Automatically groups sequences by prefix (e.g., GCA_947662875.1_04372 and GCA_947662875.1_03910 are in the same group)
- Only calculates similarities within matching groups
- Marks the top scoring key for each lock
- Outputs CSV with columns: group, lock_id, key_id, similarity, is_top_score

### Command Line Arguments

- `--locks_file`: Path to locks embeddings file (required)
- `--keys_file`: Path to keys embeddings file (required)
- `--output_file`: Path to save similarity scores CSV (required)
- `--top_k`: Number of top scores to mark per lock (default: 1)

## UMAP Visualization

Visualize protein embeddings using UMAP dimensionality reduction, colored by lock/key sets or sequence groups.

### Usage

```bash
# Basic visualization colored by lock/key
python visualize_embeddings.py --embeddings locks.npz keys.npz --labels locks keys --output umap_plot.png

# Include group-based coloring
python visualize_embeddings.py --embeddings locks.npz keys.npz --labels locks keys --output umap_plot.png --group_by_prefix

# Create grid visualization with individual group subplots
python visualize_embeddings.py --embeddings locks.npz keys.npz --labels locks keys --output umap_plot.png --grid_subplots
```

### Features

- UMAP dimensionality reduction for visualization
- Color coding by lock/key labels (locks as squares, keys as circles)
- Optional coloring by sequence group prefix
- Grid visualization mode showing first 36 groups individually
- Exports UMAP coordinates to CSV
- Customizable UMAP parameters

### Command Line Arguments

- `--embeddings`: Paths to embedding files (space-separated)
- `--labels`: Labels for each embedding file (e.g., locks keys)
- `--output`: Output plot file (supports png, pdf, svg)
- `--n_neighbors`: UMAP n_neighbors parameter (default: 15)
- `--min_dist`: UMAP min_dist parameter (default: 0.1)
- `--metric`: Distance metric for UMAP (default: cosine)
- `--group_by_prefix`: Create additional plot colored by group prefix
- `--grid_subplots`: Create 6x6 grid showing individual groups
- `--figsize`: Figure size as width height (default: 12 8)

## Filter Single Pairs

Filter datasets to include only groups with exactly one lock and one key sequence.

### Usage

```bash
python filter_single_pairs.py --locks_file locks.tsv --keys_file keys.tsv --output_locks filtered_locks.tsv --output_keys filtered_keys.tsv
```

### Features

- Identifies groups by sequence ID prefix (everything before last underscore)
- Filters to keep only groups with exactly 1 lock and 1 key
- Preserves original file format
- Provides statistics on filtering results

### Command Line Arguments

- `--locks_file`: Path to locks TSV file (required)
- `--keys_file`: Path to keys TSV file (required)
- `--output_locks`: Path for filtered locks output (required)
- `--output_keys`: Path for filtered keys output (required)

## License

MIT License
