import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def extract_clusters_from_umap(coords_file, output_dir, method='dbscan', 
                              eps=0.5, min_samples=5, n_clusters=None):
    """Extract clusters from UMAP coordinates and save cluster assignments"""
    
    # Load UMAP coordinates
    print(f"Loading UMAP coordinates from {coords_file}")
    df = pd.read_csv(coords_file)
    
    # Extract coordinates
    coords = df[['umap_1', 'umap_2']].values
    
    # Perform clustering
    if method == 'dbscan':
        print(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})")
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords)
    else:  # kmeans
        if n_clusters is None:
            # Find optimal number of clusters using silhouette score
            silhouette_scores = []
            K = range(2, min(10, len(df) // 10))
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(coords)
                score = silhouette_score(coords, labels)
                silhouette_scores.append(score)
            
            n_clusters = K[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {n_clusters}")
        
        print(f"Performing K-means clustering (n_clusters={n_clusters})")
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clustering.fit_predict(coords)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full dataframe with clusters
    full_output = output_path / 'clustered_data.csv'
    df.to_csv(full_output, index=False)
    print(f"\nFull clustered data saved to {full_output}")
    
    # Analyze clusters by label (locks vs keys)
    print("\nCluster composition:")
    cluster_summary = df.groupby(['cluster', 'label']).size().unstack(fill_value=0)
    print(cluster_summary)
    
    # Save separate files for each cluster
    unique_clusters = sorted(df['cluster'].unique())
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # DBSCAN noise points
            cluster_name = 'noise'
        else:
            cluster_name = f'cluster_{cluster_id}'
        
        cluster_df = df[df['cluster'] == cluster_id]
        
        # Save all sequences in this cluster
        cluster_file = output_path / f'{cluster_name}_all.txt'
        with open(cluster_file, 'w') as f:
            for _, row in cluster_df.iterrows():
                f.write(f"{row['seq_id']}\n")
        
        # Save locks and keys separately
        for label in cluster_df['label'].unique():
            label_df = cluster_df[cluster_df['label'] == label]
            if len(label_df) > 0:
                label_file = output_path / f'{cluster_name}_{label}.txt'
                with open(label_file, 'w') as f:
                    for _, row in label_df.iterrows():
                        f.write(f"{row['seq_id']}\n")
                print(f"  {cluster_name} {label}: {len(label_df)} sequences saved")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Use different markers for locks and keys
    markers = {'locks': 's', 'keys': 'o'}
    
    # Plot each cluster
    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        
        if cluster_id == -1:
            color = 'gray'
            alpha = 0.3
        else:
            color = None  # Let matplotlib choose
            alpha = 0.6
        
        for label in cluster_df['label'].unique():
            label_df = cluster_df[cluster_df['label'] == label]
            plt.scatter(label_df['umap_1'], label_df['umap_2'], 
                       marker=markers.get(label, 'o'),
                       c=[color] if color else None,
                       label=f'Cluster {cluster_id} - {label}' if cluster_id != -1 else f'Noise - {label}',
                       alpha=alpha, s=50)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'UMAP with {method.upper()} Clusters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_path / 'cluster_visualization.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nCluster visualization saved to {plot_file}")
    
    # Print statistics
    print("\nCluster statistics:")
    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        n_locks = len(cluster_df[cluster_df['label'] == 'locks'])
        n_keys = len(cluster_df[cluster_df['label'] == 'keys'])
        cluster_name = 'Noise' if cluster_id == -1 else f'Cluster {cluster_id}'
        print(f"{cluster_name}: {len(cluster_df)} total ({n_locks} locks, {n_keys} keys)")
        
        # Show example sequence IDs
        print(f"  Example sequences: {', '.join(cluster_df['seq_id'].head(3).tolist())}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Extract clusters from UMAP coordinates')
    parser.add_argument('--coords_file', type=str, required=True, 
                       help='Path to UMAP coordinates CSV file (from visualize_embeddings.py)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save cluster results')
    parser.add_argument('--method', type=str, choices=['dbscan', 'kmeans'], default='dbscan',
                       help='Clustering method to use')
    parser.add_argument('--eps', type=float, default=0.5,
                       help='DBSCAN epsilon parameter (maximum distance between points in cluster)')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='DBSCAN minimum samples in cluster')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters for K-means (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    extract_clusters_from_umap(
        args.coords_file,
        args.output_dir,
        method=args.method,
        eps=args.eps,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters
    )

if __name__ == "__main__":
    main()