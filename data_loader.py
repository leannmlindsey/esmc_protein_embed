import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class ProteinSequenceDataset(Dataset):
    def __init__(self, file_path: str, sequence_col: str = 'sequence', id_col: str = 'id'):
        self.df = pd.read_csv(file_path, sep='\t')
        self.sequences = self.df[sequence_col].tolist()
        self.ids = self.df[id_col].tolist() if id_col in self.df.columns else list(range(len(self.sequences)))
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.ids[idx], self.sequences[idx]

def create_protein_dataloader(file_path: str, batch_size: int = 32, sequence_col: str = 'sequence', id_col: str = 'id') -> DataLoader:
    dataset = ProteinSequenceDataset(file_path, sequence_col, id_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)