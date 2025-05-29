import argparse
import numpy as np
import pickle
from pathlib import Path
from data_loader import create_protein_dataloader
from esm_embedder import ESMCEmbedder

def main():
    parser = argparse.ArgumentParser(description='Generate ESM-C embeddings for protein sequences')
    parser.add_argument('--input_file', type=str, required=True, help='Path to tab-delimited input file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save embeddings')
    parser.add_argument('--sequence_col', type=str, default='sequence', help='Column name for sequences')
    parser.add_argument('--id_col', type=str, default='id', help='Column name for sequence IDs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--format', type=str, choices=['npz', 'pkl', 'npy'], default='npz', help='Output format')
    parser.add_argument('--model', type=str, default='esmc_300m', 
                       help='ESM-C model to use: esmc_300m, esmc_600m, or HuggingFace model ID (default: esmc_300m)')
    
    args = parser.parse_args()
    
    print(f"Loading sequences from {args.input_file}")
    dataloader = create_protein_dataloader(
        args.input_file, 
        batch_size=args.batch_size,
        sequence_col=args.sequence_col,
        id_col=args.id_col
    )
    
    print(f"Initializing ESM-C model: {args.model}")
    embedder = ESMCEmbedder(model_name=args.model)
    
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