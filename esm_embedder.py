import torch
from transformers import AutoTokenizer, EsmModel
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

class ESMCEmbedder:
    def __init__(self, model_name: str = "EvolutionaryScale/esmc-300m-2024-12", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def embed_sequences(self, sequences: List[Tuple[str, str]], batch_size: int = 8) -> Dict[str, np.ndarray]:
        embeddings = {}
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch = sequences[i:i + batch_size]
            seq_ids = [item[0] for item in batch]
            seq_strs = [item[1] for item in batch]
            
            with torch.no_grad():
                inputs = self.tokenizer(seq_strs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                for j, seq_id in enumerate(seq_ids):
                    seq_len = (inputs['attention_mask'][j] == 1).sum()
                    seq_embedding = outputs.last_hidden_state[j, :seq_len].cpu().numpy()
                    embeddings[seq_id] = seq_embedding.mean(axis=0)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.model.config.hidden_size