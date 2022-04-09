from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple, Any, Dict
from tape import TAPETokenizer
import torch
import numpy as np
from torch.utils.data import DataLoader
from prosittransformer.DataHandler import pad_sequences
from tape import ProteinBertForValuePredictionFragmentationProsit


class DataframeDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
            
        data = pd.read_csv(data_file, sep=",")
        self._data = data.to_dict('records')
        
        self._num_examples = len(self._data)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = self._data[index]
        return item


CHARGES = [1, 2, 3, 4, 5, 6]
def get_precursor_charge_onehot(charges):
    array = np.zeros([len(charges), max(CHARGES)], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array

class PrositInputDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac'
                ):
        
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        self.data = DataframeDataset(data_path)
        self.keys = [
                     'modified_sequence',
                     'collision_energy',
                     'precursor_charge'
                     ]
    
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['modified_sequence'])
        input_mask = np.ones_like(token_ids)
        collision_energy = item['collision_energy'] / 100
        charge = item['precursor_charge']
        return (token_ids, input_mask, collision_energy, charge)

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, collision_energy, charge = tuple(zip(*batch))
        charge = get_precursor_charge_onehot(charge)

        collision_energy = np.stack(collision_energy)
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))

        collision_energy_tensor = torch.FloatTensor(collision_energy)
        charge_tensor = torch.FloatTensor(charge)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'collision_energy': collision_energy_tensor,
                'charge': charge_tensor}