import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random

def get_group_prefix(seq_id):
    """Extract group prefix from sequence ID (everything before last underscore)"""
    parts = seq_id.rsplit('_', 1)
    if len(parts) > 1:
        return parts[0]
    return seq_id

def filter_by_genomes(locks_file, keys_file, output_locks, output_keys, 
                      n_genomes=None, genome_list=None, random_seed=42):
    """Filter datasets to include all locks and keys from selected genomes"""
    
    # Read the files
    print(f"Reading locks from {locks_file}")
    locks_df = pd.read_csv(locks_file, sep='\t', dtype=str)
    print(f"Found {len(locks_df)} lock sequences")
    
    print(f"Reading keys from {keys_file}")
    keys_df = pd.read_csv(keys_file, sep='\t', dtype=str)
    print(f"Found {len(keys_df)} key sequences")
    
    # Group sequences by prefix
    locks_by_group = defaultdict(list)
    keys_by_group = defaultdict(list)
    
    # Assuming 'id' column contains the sequence IDs
    id_col = 'id'
    if id_col not in locks_df.columns:
        raise ValueError(f"Column '{id_col}' not found in locks file. Available columns: {list(locks_df.columns)}")
    if id_col not in keys_df.columns:
        raise ValueError(f"Column '{id_col}' not found in keys file. Available columns: {list(keys_df.columns)}")
    
    # Group locks
    for idx, row in locks_df.iterrows():
        seq_id = str(row[id_col])
        group = get_group_prefix(seq_id)
        locks_by_group[group].append(idx)
    
    # Group keys
    for idx, row in keys_df.iterrows():
        seq_id = str(row[id_col])
        group = get_group_prefix(seq_id)
        keys_by_group[group].append(idx)
    
    # Find all groups that have at least one lock or key
    all_groups = set(locks_by_group.keys()) | set(keys_by_group.keys())
    print(f"\nFound {len(all_groups)} unique genomes/groups")
    
    # Select genomes
    if genome_list:
        # Use provided list of genomes
        selected_genomes = [g for g in genome_list if g in all_groups]
        missing_genomes = [g for g in genome_list if g not in all_groups]
        if missing_genomes:
            print(f"Warning: The following genomes were not found: {missing_genomes}")
    elif n_genomes:
        # Randomly select n genomes
        random.seed(random_seed)
        available_genomes = list(all_groups)
        n_to_select = min(n_genomes, len(available_genomes))
        selected_genomes = random.sample(available_genomes, n_to_select)
    else:
        raise ValueError("Must specify either n_genomes or genome_list")
    
    print(f"\nSelected {len(selected_genomes)} genomes")
    
    # Collect indices for selected genomes
    filtered_locks_indices = []
    filtered_keys_indices = []
    
    # Statistics
    genome_stats = []
    
    for genome in sorted(selected_genomes):
        n_locks = len(locks_by_group.get(genome, []))
        n_keys = len(keys_by_group.get(genome, []))
        
        if genome in locks_by_group:
            filtered_locks_indices.extend(locks_by_group[genome])
        if genome in keys_by_group:
            filtered_keys_indices.extend(keys_by_group[genome])
        
        genome_stats.append({
            'genome': genome,
            'n_locks': n_locks,
            'n_keys': n_keys,
            'total': n_locks + n_keys
        })
    
    # Create filtered dataframes
    filtered_locks_df = locks_df.iloc[filtered_locks_indices]
    filtered_keys_df = keys_df.iloc[filtered_keys_indices]
    
    # Save filtered datasets
    output_locks_path = Path(output_locks)
    output_keys_path = Path(output_keys)
    
    output_locks_path.parent.mkdir(parents=True, exist_ok=True)
    output_keys_path.parent.mkdir(parents=True, exist_ok=True)
    
    filtered_locks_df.to_csv(output_locks_path, sep='\t', index=False)
    filtered_keys_df.to_csv(output_keys_path, sep='\t', index=False)
    
    print(f"\nFiltered datasets saved:")
    print(f"  Locks: {output_locks_path} ({len(filtered_locks_df)} sequences)")
    print(f"  Keys: {output_keys_path} ({len(filtered_keys_df)} sequences)")
    
    # Save genome list and statistics
    genome_list_file = output_locks_path.parent / 'selected_genomes.txt'
    with open(genome_list_file, 'w') as f:
        for genome in sorted(selected_genomes):
            f.write(f"{genome}\n")
    print(f"  Genome list: {genome_list_file}")
    
    # Save statistics
    stats_df = pd.DataFrame(genome_stats)
    stats_file = output_locks_path.parent / 'genome_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"  Statistics: {stats_file}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"Original locks: {len(locks_df)}")
    print(f"Original keys: {len(keys_df)}")
    print(f"Filtered locks: {len(filtered_locks_df)}")
    print(f"Filtered keys: {len(filtered_keys_df)}")
    
    # Show statistics for first few genomes
    print(f"\nPer-genome statistics (first 10):")
    print(f"{'Genome':<30} {'Locks':>8} {'Keys':>8} {'Total':>8}")
    print("-" * 56)
    for stat in genome_stats[:10]:
        print(f"{stat['genome']:<30} {stat['n_locks']:>8} {stat['n_keys']:>8} {stat['total']:>8}")
    if len(genome_stats) > 10:
        print(f"... and {len(genome_stats) - 10} more genomes")
    
    # Summary of distribution
    total_locks = sum(s['n_locks'] for s in genome_stats)
    total_keys = sum(s['n_keys'] for s in genome_stats)
    avg_locks = total_locks / len(selected_genomes) if selected_genomes else 0
    avg_keys = total_keys / len(selected_genomes) if selected_genomes else 0
    
    print(f"\nAverage per genome:")
    print(f"  Locks: {avg_locks:.1f}")
    print(f"  Keys: {avg_keys:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Filter protein datasets by selecting specific genomes')
    parser.add_argument('--locks_file', type=str, required=True, help='Path to locks TSV file')
    parser.add_argument('--keys_file', type=str, required=True, help='Path to keys TSV file')
    parser.add_argument('--output_locks', type=str, required=True, help='Path for filtered locks output')
    parser.add_argument('--output_keys', type=str, required=True, help='Path for filtered keys output')
    
    # Genome selection options (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument('--n_genomes', type=int, help='Number of genomes to randomly select')
    selection_group.add_argument('--genome_list', nargs='+', help='Specific genomes to select')
    selection_group.add_argument('--genome_file', type=str, help='File containing genome IDs (one per line)')
    
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for genome selection')
    
    args = parser.parse_args()
    
    # Handle genome file input
    genome_list = args.genome_list
    if args.genome_file:
        with open(args.genome_file, 'r') as f:
            genome_list = [line.strip() for line in f if line.strip()]
    
    filter_by_genomes(
        args.locks_file, 
        args.keys_file, 
        args.output_locks, 
        args.output_keys,
        n_genomes=args.n_genomes,
        genome_list=genome_list,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()