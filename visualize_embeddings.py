import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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
    """Extract group prefix from sequence ID"""
    parts = seq_id.rsplit('_', 1)
    if len(parts) > 1:
        return parts[0]
    return seq_id

def create_umap_visualization(embeddings_files, labels, output_file, 
                            n_neighbors=15, min_dist=0.1, metric='cosine',
                            group_by_prefix=False, figsize=(12, 8), grid_subplots=False):
    """Create UMAP visualization of embeddings"""
    
    # Load all embeddings
    all_embeddings = []
    all_ids = []
    all_labels = []
    all_groups = []
    
    for file_path, label in zip(embeddings_files, labels):
        print(f"Loading {label} embeddings from {file_path}")
        embeddings = load_embeddings(file_path)
        
        for seq_id, embedding in embeddings.items():
            all_embeddings.append(embedding)
            all_ids.append(seq_id)
            all_labels.append(label)
            all_groups.append(get_group_prefix(seq_id))
    
    # Convert to numpy array
    X = np.vstack(all_embeddings)
    print(f"Total embeddings: {X.shape[0]}, Dimensions: {X.shape[1]}")
    
    # Perform UMAP
    print("Running UMAP...")
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embeddings_2d = umap.fit_transform(X)
    
    # If grid_subplots is requested, create the grid visualization
    if grid_subplots:
        # Create main figure with all points
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        
        unique_labels = list(set(all_labels))
        color_palette = sns.color_palette("husl", len(unique_labels))
        colors = {label: color_palette[i] for i, label in enumerate(unique_labels)}
        markers = {'locks': 's', 'keys': 'o'}  # Square for locks, circle for keys
        
        for label in unique_labels:
            mask = [l == label for l in all_labels]
            points = embeddings_2d[mask]
            ax1.scatter(points[:, 0], points[:, 1], c=[colors[label]], 
                       marker=markers.get(label, 'o'), label=label, alpha=0.6, s=50)
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('UMAP Visualization - All Lock/Key Pairs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Save main figure
        main_output = Path(output_file).with_suffix('.main.png')
        plt.tight_layout()
        plt.savefig(main_output, dpi=300, bbox_inches='tight')
        print(f"Main visualization saved to {main_output}")
        plt.close()
        
        # Create 3x3 grid of individual groups
        unique_groups = sorted(list(set(all_groups)))[:9]  # Take first 9 groups
        n_groups_to_plot = min(9, len(unique_groups))
        
        fig2, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Get the global UMAP coordinate ranges for consistent axes
        x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
        y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        for idx, group in enumerate(unique_groups[:n_groups_to_plot]):
            ax = axes[idx]
            
            # Get points for this group
            group_mask = [g == group for g in all_groups]
            group_indices = [i for i, m in enumerate(group_mask) if m]
            
            # Plot only this group's points, but maintain global coordinate system
            for i in group_indices:
                label = all_labels[i]
                point = embeddings_2d[i]
                ax.scatter(point[0], point[1], c=[colors[label]],
                          marker=markers.get(label, 'o'), alpha=0.8, s=150)
            
            # Set consistent axis limits to match global UMAP
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            ax.set_title(f'{group}', fontsize=10, fontweight='bold')
            ax.set_xlabel('UMAP 1', fontsize=8)
            ax.set_ylabel('UMAP 2', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if idx == 0:
                # Create dummy points for legend
                for label in unique_labels:
                    ax.scatter([], [], c=[colors[label]], marker=markers.get(label, 'o'), 
                             label=label, s=100)
                ax.legend(fontsize=8, loc='upper right')
        
        # Hide empty subplots
        for idx in range(n_groups_to_plot, 9):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        grid_output = Path(output_file).with_suffix('.grid.png')
        plt.savefig(grid_output, dpi=300, bbox_inches='tight')
        print(f"Grid visualization saved to {grid_output}")
        plt.close()
        
    else:
        # Original visualization code
        fig, axes = plt.subplots(1, 2 if group_by_prefix else 1, figsize=figsize)
        if not group_by_prefix:
            axes = [axes]
        
        # Plot 1: Color by lock/key label
        ax = axes[0]
        unique_labels = list(set(all_labels))
        colors = sns.color_palette("husl", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in all_labels]
            points = embeddings_2d[mask]
            ax.scatter(points[:, 0], points[:, 1], c=[colors[i]], label=label, alpha=0.6, s=50)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Visualization - Colored by Lock/Key')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Color by group prefix (if requested)
        if group_by_prefix:
            ax = axes[1]
            unique_groups = list(set(all_groups))
            n_groups = len(unique_groups)
            
            # Use different color palette for groups
            if n_groups <= 20:
                colors = sns.color_palette("tab20", n_groups)
            else:
                colors = sns.color_palette("husl", n_groups)
            
            group_to_color = {group: colors[i] for i, group in enumerate(unique_groups)}
            
            # Plot each group
            for group in unique_groups:
                mask = [g == group for g in all_groups]
                points = embeddings_2d[mask]
                ax.scatter(points[:, 0], points[:, 1], c=[group_to_color[group]], 
                          label=group if n_groups <= 10 else '', alpha=0.6, s=50)
            
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP Visualization - Colored by Group Prefix')
            if n_groups <= 10:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    # Save UMAP coordinates
    coords_file = Path(output_file).with_suffix('.coords.csv')
    df = pd.DataFrame({
        'seq_id': all_ids,
        'label': all_labels,
        'group': all_groups,
        'umap_1': embeddings_2d[:, 0],
        'umap_2': embeddings_2d[:, 1]
    })
    df.to_csv(coords_file, index=False)
    print(f"UMAP coordinates saved to {coords_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total points: {len(all_ids)}")
    print(f"Unique groups: {len(set(all_groups))}")
    for label in unique_labels:
        count = sum(1 for l in all_labels if l == label)
        print(f"{label}: {count} sequences")

def main():
    parser = argparse.ArgumentParser(description='Create UMAP visualization of protein embeddings')
    parser.add_argument('--embeddings', nargs='+', required=True, help='Paths to embedding files')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each embedding file (e.g., locks keys)')
    parser.add_argument('--output', type=str, required=True, help='Output plot file (png, pdf, svg)')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='cosine', help='Distance metric for UMAP')
    parser.add_argument('--group_by_prefix', action='store_true', help='Also create plot colored by group prefix')
    parser.add_argument('--grid_subplots', action='store_true', help='Create grid of individual group subplots')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8], help='Figure size (width height)')
    
    args = parser.parse_args()
    
    if len(args.embeddings) != len(args.labels):
        raise ValueError("Number of embedding files must match number of labels")
    
    create_umap_visualization(
        args.embeddings,
        args.labels,
        args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        group_by_prefix=args.group_by_prefix,
        grid_subplots=args.grid_subplots,
        figsize=tuple(args.figsize)
    )

if __name__ == "__main__":
    main()