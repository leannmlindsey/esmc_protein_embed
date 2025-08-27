# Protein Embedding Models

This repository now supports three families of ESM (Evolutionary Scale Modeling) protein language models:

## Supported Models

### ESM-C (ESM Comprehensive)
- `esmc_300m` - 300M parameters (default for ESM-C)
- `esmc_600m` - 600M parameters

### ESM-2
- `esm2_t48_15B` - 15B parameters
- `esm2_t36_3B` - 3B parameters  
- `esm2_t33_650M` - 650M parameters (default for ESM-2)
- `esm2_t30_150M` - 150M parameters
- `esm2_t12_35M` - 35M parameters
- `esm2_t6_8M` - 8M parameters

### ESM-1b
- `esm1b_t33_650M` - 650M parameters (default for ESM-1b)

## Usage

### Basic Usage

Generate embeddings using the default model for each type:

```bash
# ESM-C (default)
python src/embed_proteins.py --input_file proteins.tsv --output_file embeddings.npz

# ESM-2
python src/embed_proteins.py --model_type esm2 --input_file proteins.tsv --output_file embeddings.npz

# ESM-1b
python src/embed_proteins.py --model_type esm1b --input_file proteins.tsv --output_file embeddings.npz
```

### Specify Model Variant

Choose a specific model variant:

```bash
# Use ESM-2 with 3B parameters
python src/embed_proteins.py --model_type esm2 --model esm2_t36_3B --input_file proteins.tsv --output_file embeddings.npz

# Use ESM-C with 600M parameters
python src/embed_proteins.py --model_type esmc --model esmc_600m --input_file proteins.tsv --output_file embeddings.npz
```

### Command Line Arguments

- `--model_type`: Choose between `esmc`, `esm2`, or `esm1b` (default: `esmc`)
- `--model`: Specific model variant or `auto` for default (default: `auto`)
- `--input_file`: Path to tab-delimited input file with sequences
- `--output_file`: Path to save embeddings
- `--sequence_col`: Column name for sequences (default: `sequence`)
- `--id_col`: Column name for sequence IDs (default: `id`)
- `--batch_size`: Batch size for processing (default: 8)
- `--format`: Output format - `npz`, `pkl`, or `npy` (default: `npz`)

## Model Selection Guide

### When to use ESM-C
- Latest and most comprehensive model
- Best for general-purpose protein embeddings
- Includes structural information

### When to use ESM-2
- Well-established model with multiple size options
- Good balance between performance and computational requirements
- Choose model size based on available resources

### When to use ESM-1b
- Legacy compatibility
- Smaller memory footprint
- Well-validated on many downstream tasks

## Testing

Run the test script to verify all models are working:

```bash
python test_embedders.py
```

## Requirements

The implementation requires:
- `torch`
- `esm` (fair-esm package)
- `numpy`
- `tqdm`

Install with:
```bash
pip install torch fair-esm numpy tqdm
```