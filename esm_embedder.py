import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

class ESMCEmbedder:
    def __init__(self, model_name: str = "esmc_300m", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading ESM-C model on {self.device}...")
        self.model = ESMC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def embed_sequences(self, sequences: List[Tuple[str, str]], batch_size: int = 8) -> Dict[str, np.ndarray]:
        embeddings = {}
        
        # Process sequences one at a time (ESM-C API requirement)
        for seq_id, seq_str in tqdm(sequences, desc="Generating embeddings"):
            try:
                # Ensure sequence is a string
                if not isinstance(seq_str, str):
                    seq_str = str(seq_str)
                
                # Clean sequence - remove any whitespace and ensure uppercase
                seq_str = seq_str.strip().upper()
                
                # Validate sequence contains only valid amino acids
                valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                if not all(aa in valid_aa for aa in seq_str):
                    print(f"Warning: Sequence {seq_id} contains non-standard amino acids, skipping")
                    continue
                
                with torch.no_grad():
                    # Create ESMProtein object
                    protein = ESMProtein(sequence=seq_str)
                    
                    # Encode protein to tensor representation
                    protein_tensor = self.model.encode(protein)
                    
                    # Get embeddings through logits with embedding flag
                    logits_output = self.model.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True)
                    )
                    
                    # Extract embeddings
                    embedding = logits_output.embeddings
                    
                    # Convert to numpy and average pool
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    
                    # Handle different tensor shapes
                    if len(embedding.shape) == 3:  # [batch, seq_len, hidden_dim]
                        embedding = embedding[0].mean(axis=0)
                    elif len(embedding.shape) == 2:  # [seq_len, hidden_dim]
                        embedding = embedding.mean(axis=0)
                    
                    embeddings[seq_id] = embedding
                    
            except Exception as e:
                print(f"Error processing sequence {seq_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        # ESM-C 300M has 960 hidden dimensions
        return 960