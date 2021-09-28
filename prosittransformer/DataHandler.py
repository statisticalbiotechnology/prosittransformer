from tape.datasets import PrositFragmentationDataset
from typing import List, Union, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

import multiprocessing
from typing import List, Tuple, Any, Sequence, Dict
from .constants import MAX_LENGTH
from pathlib import Path

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    """Pad batches so all datapoints have the same length"""
    batch_size = len(sequences)
    ## We have to set fixed length for all batches, otherwise TF will be sloooooow.
    #Hard coded
    shape = [batch_size] + [MAX_LENGTH]
    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

#We need to overwrite collate_fn to assert that all batches has the same length, otherwise TF will be slow.
class DATASET(PrositFragmentationDataset):
    def __init__(self, lmdb: Path, data_type: str, returnTorchTensor: bool = False)->None:
        PrositFragmentationDataset.__init__(self, lmdb, data_type)
        self.returnTorchTensor = returnTorchTensor

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """Prepare batch"""
        input_ids, input_mask, intensities_raw_true_value, collision_energy, charge = tuple(zip(*batch))
        collision_energy = np.stack(collision_energy)
        input_ids = pad_sequences(input_ids, 0)
        input_mask = pad_sequences(input_mask, 0)
        if self.returnTorchTensor:
            input_ids = torch.from_numpy(input_ids)
            input_mask = torch.from_numpy(input_mask)
            intensities_raw_true_value = torch.FloatTensor(intensities_raw_true_value)  # type: ignore

            collision_energy = torch.FloatTensor(collision_energy)
            charge = torch.FloatTensor(charge)

        #Input mask must be last, otherwise conversion fails!
        return {'input_ids': input_ids,
                'collision_energy': collision_energy,
                'charge': charge,
                'input_mask': input_mask}

def getTFDataLoader(lmdb: Path, split: str, num_workers : int =multiprocessing.cpu_count()-1, batch_size: int = 64, returnTorchTensor: bool = False)->DataLoader:
    """Get dataloader for TF"""
    dataset = DATASET(lmdb, split, returnTorchTensor)
    return DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,  # type: ignore
        batch_size=batch_size)

def getTorchDataLoader(lmdb: Path, split: str, num_workers:int=multiprocessing.cpu_count()-1, batch_size:int=64)->DataLoader:
    """Get dataloader for Torch"""
    dataset = PrositFragmentationDataset(lmdb, split)
    return DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,  # type: ignore
        batch_size=batch_size)

def loadNPY(path : str, file: str, returnPath : bool)->np.array:
    """Load .npy file"""
    if not path.endswith("/"):
        path += "/"
    path = f"{path}{file}"
    if returnPath:
        return np.load(path), path
    else:
        return np.load(path)