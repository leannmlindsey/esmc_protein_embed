import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict

def load_embeddings(file_path):
    """Load embeddings from npz, pkl, or npy format"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.npy':
        embeddings = np.load(file_path)
        ids_file = file_path.with_suffix('.ids.txt')
        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f]
        return {ids[i]: embeddings[i] for i in range(len(ids))}
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def get_group_prefix(seq_id):
    """Extract group prefix from sequence ID (everything before last underscore)"""
    parts = seq_id.rsplit('_', 1)
    if len(parts) > 1:
        return parts[0]
    return seq_id

def group_embeddings_by_prefix(embeddings):
    """Group embeddings by their prefix"""
    grouped = defaultdict(dict)
    for seq_id, embedding in embeddings.items():
        prefix = get_group_prefix(seq_id)
        grouped[prefix][seq_id] = embedding
    return grouped

def calculate_similarities_for_group(lock_embeddings, key_embeddings, group_name):
    """Calculate cosine similarities between locks and keys in a single group"""
    
    lock_ids = list(lock_embeddings.keys())
    key_ids = list(key_embeddings.keys())
    
    # Stack embeddings into matrices
    locks_matrix = np.vstack([lock_embeddings[lid] for lid in lock_ids])
    keys_matrix = np.vstack([key_embeddings[kid] for kid in key_ids])
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(locks_matrix, keys_matrix)
    
    # Create results for this group
    results = []
    
    # For each lock, find all similarities and mark the top one
    for i, lock_id in enumerate(lock_ids):
        similarities = similarity_matrix[i]
        top_idx = np.argmax(similarities)
        
        for j, key_id in enumerate(key_ids):
            results.append({
                'group': group_name,
                'lock_id': lock_id,
                'key_id': key_id,
                'similarity': similarities[j],
                'is_top_score': j == top_idx
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate cosine similarities between lock and key protein embeddings grouped by prefix')
    parser.add_argument('--locks_file', type=str, required=True, help='Path to locks embeddings file')
    parser.add_argument('--keys_file', type=str, required=True, help='Path to keys embeddings file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save similarity scores CSV')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top scores to mark (default: 1)')
    
    args = parser.parse_args()
    
    print(f"Loading lock embeddings from {args.locks_file}")
    locks_embeddings = load_embeddings(args.locks_file)
    print(f"Loaded {len(locks_embeddings)} lock embeddings")
    
    print(f"Loading key embeddings from {args.keys_file}")
    keys_embeddings = load_embeddings(args.keys_file)
    print(f"Loaded {len(keys_embeddings)} key embeddings")
    
    # Group embeddings by prefix
    print("Grouping embeddings by prefix...")
    grouped_locks = group_embeddings_by_prefix(locks_embeddings)
    grouped_keys = group_embeddings_by_prefix(keys_embeddings)
    
    # Find common groups
    common_groups = set(grouped_locks.keys()) & set(grouped_keys.keys())
    print(f"Found {len(common_groups)} common groups")
    
    # Calculate similarities for each group
    all_results = []
    for group in tqdm(sorted(common_groups), desc="Processing groups"):
        group_locks = grouped_locks[group]
        group_keys = grouped_keys[group]
        
        if group_locks and group_keys:  # Only process if both have entries
            group_results = calculate_similarities_for_group(group_locks, group_keys, group)
            all_results.extend(group_results)
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    # If top_k > 1, update the is_top_score column
    if args.top_k > 1 and len(results_df) > 0:
        print(f"Marking top {args.top_k} scores for each lock...")
        results_df['is_top_score'] = False
        for (group, lock_id) in results_df.groupby(['group', 'lock_id']).groups:
            mask = (results_df['group'] == group) & (results_df['lock_id'] == lock_id)
            lock_df = results_df[mask]
            top_indices = lock_df.nlargest(min(args.top_k, len(lock_df)), 'similarity').index
            results_df.loc[top_indices, 'is_top_score'] = True
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nSimilarity scores saved to {output_path}")
    print(f"Total comparisons: {len(results_df)}")
    if len(results_df) > 0:
        print(f"Top scores marked: {results_df['is_top_score'].sum()}")
        
        # Print summary statistics
        print("\nSummary statistics:")
        print(f"Mean similarity: {results_df['similarity'].mean():.4f}")
        print(f"Min similarity: {results_df['similarity'].min():.4f}")
        print(f"Max similarity: {results_df['similarity'].max():.4f}")
        
        # Print per-group statistics
        print("\nPer-group statistics:")
        group_stats = results_df.groupby('group').agg({
            'similarity': ['count', 'mean', 'min', 'max'],
            'lock_id': 'nunique',
            'key_id': 'nunique'
        })
        group_stats.columns = ['comparisons', 'mean_sim', 'min_sim', 'max_sim', 'n_locks', 'n_keys']
        print(group_stats.head(10))
        if len(group_stats) > 10:
            print(f"... and {len(group_stats) - 10} more groups")

if __name__ == "__main__":
    main()