import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

class ESMCEmbedder:
    def __init__(self, model_name: str = "esmc_300m", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading ESM-C model '{model_name}' on {self.device}...")
        
        # Map common names to full HuggingFace IDs if needed
        model_mapping = {
            'esmc_300m': 'esmc_300m',
            'esmc_600m': 'esmc_600m',
            'esmc-300m': 'esmc_300m',
            'esmc-600m': 'esmc_600m',
        }
        
        # Use mapping if available, otherwise use the provided name directly
        actual_model = model_mapping.get(model_name, model_name)
        
        try:
            self.model = ESMC.from_pretrained(actual_model).to(self.device)
            self.model.eval()
            print(f"Model '{actual_model}' loaded successfully!")
        except Exception as e:
            print(f"Error loading model '{actual_model}': {str(e)}")
            print("Trying with HuggingFace model ID format...")
            # If that fails, try as a HuggingFace model ID
            try:
                self.model = ESMC.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                self.model.eval()
                print(f"Model '{model_name}' loaded successfully!")
            except Exception as e2:
                raise ValueError(f"Failed to load model '{model_name}'. Error: {str(e2)}")
    
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
                
                # Skip empty sequences
                if not seq_str:
                    print(f"Warning: Empty sequence for {seq_id}, skipping")
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
                continue
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        # Get the actual embedding dimension from the model
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model, 'd_model'):
            return self.model.d_model
        else:
            # Default dimensions for known models
            return 960  # ESM-C 300M default