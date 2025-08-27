import argparse
import numpy as np
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_loader import create_protein_dataloader
from esm_embedder import ESMCEmbedder
from esm2_embedder import ESM2Embedder
from esm1b_embedder import ESM1bEmbedder

def get_embedder(model_type, model_name):
    if model_type == 'esmc':
        return ESMCEmbedder(model_name=model_name)
    elif model_type == 'esm2':
        return ESM2Embedder(model_name=model_name)
    elif model_type == 'esm1b':
        return ESM1bEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description='Generate protein embeddings using ESM models')
    parser.add_argument('--input_file', type=str, required=True, help='Path to tab-delimited input file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save embeddings')
    parser.add_argument('--sequence_col', type=str, default='sequence', help='Column name for sequences')
    parser.add_argument('--id_col', type=str, default='id', help='Column name for sequence IDs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--format', type=str, choices=['npz', 'pkl', 'npy'], default='npz', help='Output format')
    parser.add_argument('--model_type', type=str, default='esmc', choices=['esmc', 'esm2', 'esm1b'],
                       help='Model type to use: esmc, esm2, or esm1b (default: esmc)')
    parser.add_argument('--model', type=str, default='auto',
                       help='Model variant. Use "auto" for default, or specify: '
                            'ESM-C: esmc_300m, esmc_600m; '
                            'ESM-2: esm2_t48_15B, esm2_t36_3B, esm2_t33_650M, esm2_t30_150M, esm2_t12_35M, esm2_t6_8M; '
                            'ESM-1b: esm1b_t33_650M (default: auto)')
    
    args = parser.parse_args()
    
    print(f"Loading sequences from {args.input_file}")
    dataloader = create_protein_dataloader(
        args.input_file, 
        batch_size=args.batch_size,
        sequence_col=args.sequence_col,
        id_col=args.id_col
    )
    
    # Set default model based on model type if auto is specified
    if args.model == 'auto':
        default_models = {
            'esmc': 'esmc_300m',
            'esm2': 'esm2_t33_650M_UR50D',
            'esm1b': 'esm1b_t33_650M_UR50S'
        }
        model_name = default_models[args.model_type]
    else:
        model_name = args.model
    
    print(f"Initializing {args.model_type.upper()} model: {model_name}")
    embedder = get_embedder(args.model_type, model_name)
    
    all_sequences = []
    for batch in dataloader:
        ids, sequences = batch
        all_sequences.extend(zip(ids, sequences))
    
    print(f"Processing {len(all_sequences)} sequences")
    embeddings = embedder.embed_sequences(all_sequences, batch_size=args.batch_size)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'npz':
        np.savez_compressed(output_path, **embeddings)
        print(f"Embeddings saved to {output_path} in npz format")
    elif args.format == 'pkl':
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {output_path} in pickle format")
    else:
        embeddings_array = np.array(list(embeddings.values()))
        np.save(output_path, embeddings_array)
        ids_file = output_path.with_suffix('.ids.txt')
        with open(ids_file, 'w') as f:
            for seq_id in embeddings.keys():
                f.write(f"{seq_id}\n")
        print(f"Embeddings saved to {output_path} and IDs to {ids_file}")

if __name__ == "__main__":
    main()