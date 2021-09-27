import lmdb
from tensorflow.keras.utils import HDF5Matrix
import numpy as np
from tqdm import tqdm
import os 
from typing import TypeVar, Type, Union, Dict
from ..utils import SequenceConverter, BatchLoader, SaveLMDB, hdf5Loader

T = TypeVar('T', bound='Prosit2Tape')

class Prosit2Tape(SequenceConverter, BatchLoader, SaveLMDB, hdf5Loader):
    """ Prosi-to-Tape data converter"""
    def __init__(self, split: str, data: HDF5Matrix, out_dir: str)->None:
        SequenceConverter.__init__(self)
        BatchLoader.__init__(self)
        SaveLMDB.__init__(self)
        self._keys = [
            "sequence_integer",
            "collision_energy_aligned_normed",
            "precursor_charge_onehot",
            "intensities_raw"
            ]

        assert self._keys_exists(data), "The dataset is not complete"

        self._split = split
        self._dataset = self._get_dataset(data)
        self._n_data_points = data["collision_energy_aligned_normed"].shape[0]        
        self._out_dir = out_dir
        
        if not self.isDir(self._out_dir):
            self.createDir(self._out_dir)
        
        self._out_path = self.out_path(self._split)

        print(f"Save lmbd file at {self._out_path}")
        
    def _keys_exists(self, data: HDF5Matrix)->bool:
        """ Check if all necessary keys in hdf5 file exists """
        keys = list(data.keys())
        return all([k in keys for k in self._keys])

    def _get_dataset(self, data: Dict[str, HDF5Matrix])->dict:
        """ Extract necessary data columns """
        return { k : data[k] for k in self._keys }

    @property
    def split(self)->str:
        """ Get data split train/val/test"""
        return self._split

    @property 
    def dataset(self)->HDF5Matrix:
        """ Get hdf5 data """
        return self.dataset

    @property
    def n_data_points(self)->int:
        """ Get number of datapoints """
        return self._n_data_points

    @classmethod
    def convertFromPath(cls: Type[T], split:str, path:str, out_dir:str)->T:
        """ Init for path to dataset """
        data = cls.from_hdf5(path)
        return cls(split, data, out_dir)

    def out_path(self, split:str)->str:
        """ Get out path """
        o = self._out_dir + f"/prosit_fragmentation"
        if not self.isDir(o):
            self.createDir(o)
        if split == "ho":
            split = "test"    
        _split = split
        return o + f"/prosit_fragmentation_{split}.lmdb"

    def setDataType(self, k:str, v:np.array)->Union[str, np.array]:
        """ Set type for datapoint in LMBD """
        if k == "collision_energy_aligned_normed":
            return np.array(v, dtype=np.float32)
        if k == "precursor_charge_onehot":
            return np.array(v, dtype=np.uint8)
        if k == "intensities_raw":
            return np.array(v, dtype=np.float32)
        if k == "masses_raw":
            return np.array(v, dtype=np.float32)
        if k == "sequence_integer":
            return v
       
    def convert(self, batch_size:int = 100_000)->None:
        """ Convert hdf5 to LMDB """
        self.deleteDir(self._out_path)
        self.createLMDBdir(self._out_path, self._n_data_points)

        for c, (start, end) in enumerate(self.getBatchIxs(self._n_data_points, batch_size)):
            batches = {k: self._dataset[k][start:end] for k in self._keys}

            for ix, key in enumerate(tqdm(range(start, end))):
                d = { k : self.setDataType(k, v[ix]) for k, v in batches.items()}
                d["peptide_sequence"] = self.intToPeptide(batches["sequence_integer"][ix])
                #We don't need this for Tape anymore
                del d["sequence_integer"]
                self.save(self._out_path, d, key)

class PrositNPYtoTapeLMDB(SequenceConverter, BatchLoader, SaveLMDB, hdf5Loader):
    """ Prosi-to-Tape data converter"""
    def __init__(self, split: str, X: np.ndarray, y : np.ndarray, out_dir: str)->None:
        SequenceConverter.__init__(self)
        BatchLoader.__init__(self)
        SaveLMDB.__init__(self)
        self._keys = [
            "sequence_integer",
            "iRT"
            ]

        self._split = split
        self._X = X
        self._y = y
        
        self._get_dataset(X,y)
        
        self._n_data_points = len(y)
        self._dataset = self._get_dataset(X,y)
        self._out_dir = out_dir
        
        if not self.isDir(self._out_dir):
            self.createDir(self._out_dir)
        
        self._out_path = self.out_path(self._split, out_dir)

        print(f"Save lmbd file at {self._out_path}")
    
    def _get_dataset(self, X: np.ndarray, y: np.ndarray )->dict:
        """ Extract necessary data columns """
        return { "sequence_integer" : X, "iRT": y}
    
    @property
    def split(self)->str:
        """ Get data split train/val/test"""
        return self._split

    @property 
    def X(self)->np.ndarray:
        """ Get hdf5 data """
        return self._X
    
    @property 
    def y(self)->np.ndarray:
        """ Get hdf5 data """
        return self._y

    @property
    def n_data_points(self)->int:
        """ Get number of datapoints """
        return self._n_data_points

    @property
    def out_path(self)->str:
        """ Get outpath """
        return self._out_path

    def out_path(self, split:str, o: str)->str:
        """ Get out path """
        if not self.isDir(o):
            self.createDir(o)
        _split = split
        return o + f"/prosit_iRT_{split}.lmdb"

    def setDataType(self, k:str, v:np.array)->Union[str, np.array]:
        """ Set type for datapoint in LMBD """
        if k == "iRT":
            return np.array(v, dtype=np.float32)
        if k == "sequence_integer":
            return v
       
    def convert(self, batch_size:int = 100_000)->None:
        """ Convert hdf5 to LMDB """
        self.deleteDir(self._out_path)
        self.createLMDBdir(self._out_path, self._n_data_points)

        for c, (start, end) in enumerate(self.getBatchIxs(self._n_data_points, batch_size)):
            batches = {k: self._dataset[k][start:end] for k in self._keys}

            for ix, key in enumerate(tqdm(range(start, end))):
                d = { k : self.setDataType(k, v[ix]) for k, v in batches.items()}
                d["peptide_sequence"] = self.intToPeptide(batches["sequence_integer"][ix])
                #We don't need this for Tape anymore
                del d["sequence_integer"]
                self.save(self._out_path, d, key)
         