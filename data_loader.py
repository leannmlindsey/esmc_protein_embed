import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class ProteinSequenceDataset(Dataset):
    def __init__(self, file_path: str, sequence_col: str = 'sequence', id_col: str = 'id'):
        # Read with explicit tab delimiter
        self.df = pd.read_csv(file_path, sep='\t', dtype=str, engine='python')
        
        # Debug output
        print(f"Loaded dataframe with columns: {list(self.df.columns)}")
        print(f"Number of columns: {len(self.df.columns)}")
        print(f"First few rows:")
        print(self.df.head(3))
        
        # Check if columns exist
        if sequence_col not in self.df.columns:
            raise ValueError(f"Column '{sequence_col}' not found. Available columns: {list(self.df.columns)}")
        if id_col not in self.df.columns:
            raise ValueError(f"Column '{id_col}' not found. Available columns: {list(self.df.columns)}")
        
        # Ensure sequences are strings and clean them
        self.sequences = [str(seq).strip() for seq in self.df[sequence_col].tolist()]
        self.ids = [str(id_).strip() for id_ in self.df[id_col].tolist()]
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.ids[idx], self.sequences[idx]

def create_protein_dataloader(file_path: str, batch_size: int = 32, sequence_col: str = 'sequence', id_col: str = 'id') -> DataLoader:
    dataset = ProteinSequenceDataset(file_path, sequence_col, id_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)