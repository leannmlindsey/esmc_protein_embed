import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

def get_group_prefix(seq_id):
    """Extract group prefix from sequence ID (everything before last underscore)"""
    parts = seq_id.rsplit('_', 1)
    if len(parts) > 1:
        return parts[0]
    return seq_id

def filter_single_pairs(locks_file, keys_file, output_locks, output_keys):
    """Filter datasets to only include groups with exactly one lock and one key"""
    
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
    
    # Find groups with exactly one lock and one key
    single_pair_groups = []
    for group in locks_by_group:
        if group in keys_by_group:
            if len(locks_by_group[group]) == 1 and len(keys_by_group[group]) == 1:
                single_pair_groups.append(group)
    
    print(f"\nFound {len(single_pair_groups)} groups with exactly one lock and one key")
    
    # Filter dataframes
    filtered_locks_indices = []
    filtered_keys_indices = []
    
    for group in single_pair_groups:
        filtered_locks_indices.extend(locks_by_group[group])
        filtered_keys_indices.extend(keys_by_group[group])
    
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
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Original locks: {len(locks_df)}")
    print(f"Original keys: {len(keys_df)}")
    print(f"Filtered locks: {len(filtered_locks_df)}")
    print(f"Filtered keys: {len(filtered_keys_df)}")
    print(f"Reduction: {100 * (1 - len(filtered_locks_df) / len(locks_df)):.1f}% for locks, "
          f"{100 * (1 - len(filtered_keys_df) / len(keys_df)):.1f}% for keys")
    
    # Show a few examples
    if len(single_pair_groups) > 0:
        print(f"\nExample groups (first 5):")
        for i, group in enumerate(single_pair_groups[:5]):
            lock_idx = locks_by_group[group][0]
            key_idx = keys_by_group[group][0]
            lock_id = locks_df.iloc[lock_idx][id_col]
            key_id = keys_df.iloc[key_idx][id_col]
            print(f"  Group: {group}")
            print(f"    Lock: {lock_id}")
            print(f"    Key: {key_id}")

def main():
    parser = argparse.ArgumentParser(description='Filter protein datasets to include only groups with exactly one lock and one key')
    parser.add_argument('--locks_file', type=str, required=True, help='Path to locks TSV file')
    parser.add_argument('--keys_file', type=str, required=True, help='Path to keys TSV file')
    parser.add_argument('--output_locks', type=str, required=True, help='Path for filtered locks output')
    parser.add_argument('--output_keys', type=str, required=True, help='Path for filtered keys output')
    
    args = parser.parse_args()
    
    filter_single_pairs(args.locks_file, args.keys_file, args.output_locks, args.output_keys)

if __name__ == "__main__":
    main()