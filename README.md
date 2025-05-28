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

# Install dependencies
conda env create -n esmc
pip install -r requirements.txt
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

## License

MIT License
