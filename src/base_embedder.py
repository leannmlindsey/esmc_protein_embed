from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np
import torch

class BaseEmbedder(ABC):
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def embed_sequences(self, sequences: List[Tuple[str, str]], batch_size: int = 8) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass
    
    def clean_sequence(self, seq: str) -> str:
        if not isinstance(seq, str):
            seq = str(seq)
        return seq.strip().upper()