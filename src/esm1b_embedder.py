import torch
import esm
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from base_embedder import BaseEmbedder

class ESM1bEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "esm1b_t33_650M_UR50S", device: str = None):
        super().__init__(model_name, device)
        
        self.model_mapping = {
            'esm1b': 'esm1b_t33_650M_UR50S',
            'esm1b_t33_650M': 'esm1b_t33_650M_UR50S',
        }
        
        self.load_model()
        
    def load_model(self):
        print(f"Loading ESM-1b model '{self.model_name}' on {self.device}...")
        
        actual_model = self.model_mapping.get(self.model_name, self.model_name)
        
        try:
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(actual_model)
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Model '{actual_model}' loaded successfully!")
        except Exception as e:
            raise ValueError(f"Failed to load ESM-1b model '{actual_model}'. Error: {str(e)}")
    
    def embed_sequences(self, sequences: List[Tuple[str, str]], batch_size: int = 8) -> Dict[str, np.ndarray]:
        embeddings = {}
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
            batch = sequences[i:i+batch_size]
            
            prepared_batch = []
            for seq_id, seq_str in batch:
                seq_str = self.clean_sequence(seq_str)
                if not seq_str:
                    print(f"Warning: Empty sequence for {seq_id}, skipping")
                    continue
                prepared_batch.append((seq_id, seq_str))
            
            if not prepared_batch:
                continue
                
            try:
                with torch.no_grad():
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(prepared_batch)
                    batch_tokens = batch_tokens.to(self.device)
                    
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                    token_representations = results["representations"][33]
                    
                    for j, (seq_id, seq_str) in enumerate(prepared_batch):
                        seq_len = len(seq_str)
                        embedding = token_representations[j, 1:seq_len+1].mean(0).cpu().numpy()
                        embeddings[seq_id] = embedding
                        
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                continue
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        if hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        elif hasattr(self.model, 'args') and hasattr(self.model.args, 'embed_dim'):
            return self.model.args.embed_dim
        else:
            return 1280