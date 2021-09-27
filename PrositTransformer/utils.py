import lmdb
from tensorflow.keras.utils import HDF5Matrix
import h5py
import pickle as pkl
import numpy as np
import shutil
from tqdm import tqdm
import os 
from typing import Union, Tuple, Dict, List
from pathlib import Path
import functools
from typing import Sequence
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """
    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False)->None:

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples
        
    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item

class hdf5Loader:
    """ Load hdf5 file """
    @staticmethod
    def from_hdf5(path: str, n_samples:Union[int,None] = None)->HDF5Matrix:
        # Get a list of the keys for the datasets
        with h5py.File(path, 'r') as f:
            dataset_list = list(f.keys())
        # Assemble into a dictionary
        data = dict()
        for dataset in dataset_list:
            data[dataset] = HDF5Matrix(path, dataset, start=0, end=n_samples, normalizer=None)
        return data

    @staticmethod
    def to_hdf5(dictionary : dict, path: Path)->None:
        with h5py.File(path, "w") as f:
            for key, data in dictionary.items():
                f.create_dataset(key, data=data, dtype=data.dtype, compression="gzip")

    

class SequenceConverter:
    """ Convert Prosit integer peptide sequence to Tape peptide sequence """
    @staticmethod
    def intToPeptide(sequence:np.array)->str:
        #Set M(ox) = X i.e., IUPAC unlown aminoacid
        ALPHABET = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "X": 21,
        }
        ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}
        return "".join([ALPHABET_S[int(i)] for i in sequence if int(i)!=0])

class BatchLoader:
    """ Load hdf5 in batches for speed-up"""
    def __init__(self):
        pass
    @staticmethod
    def getChunks(n: int, batch_size: int)->int:
        """ Get number of batches """
        return (n - 1) // batch_size + 1

    def getBatchIxs(self, n: int, batch_size: int)->Tuple[int,int]:
        """ Get start and end index for batch """
        chunks = self.getChunks(n, batch_size)
        for i in tqdm(range(chunks)):
            start, end = i*batch_size, (i+1)*batch_size
            if end >= n:
                end = n
            yield start, end

class PathHandler:
    """ Handled dir for data """
    def deleteDir(self, rt_path: str)->None:
        """ Delete dir """
        if self.isDir(rt_path):
            try:
                shutil.rmtree(rt_path)
            except Exception as e:
                print(e)
    def deleteFile(self, path: str)->None:
        if os.path.isfile(path):
            os.remove(path)

    @staticmethod
    def createLMDBdir(rt_path:str, n_data_points: int)->None:
        """ Creat LMDB dir """
        env = lmdb.open(str(rt_path), map_size=int(1e12))
        with env.begin(write=True) as txn:
            key = b"num_examples"
            value = n_data_points
            txn.put(key, pkl.dumps(value))
        env.close()

    @staticmethod
    def isDir(path: str)->bool:
        """ Check if dir exists """
        return os.path.isdir(path) 

    @staticmethod
    def isFile(path: str)->bool:
        """ Check if file exists """
        return Path(path).is_file()
    
    @staticmethod
    def createDir(path: str)->None:
        """ Create dir """
        os.mkdir(path)
    
class SaveLMDB(PathHandler):
    """ Save datapoint to LMDB file """
    def __init__(self):
        PathHandler.__init__(self)

    @staticmethod
    def save(rt_path: str, data: dict, key: str)->None:
        """ Save data point """
        env = lmdb.open(str(rt_path), map_size=int(1e12))
        with env.begin(write=True) as txn:
            key = str(key)
            txn.put(key.encode("ascii"), pkl.dumps(data))
        env.close()

class cleanTapeOutput:
    """Clean up Tape intensity prediction and compute spectral angle"""
    @staticmethod
    def normalize_base_peak(array: np.array)->np.array:
        """Normalize intensity peaks"""
        maxima = array.max(axis=1)
        array = array / maxima[:, np.newaxis]
        return array
    
    @staticmethod
    def mask_outofrange(array:np.array, lengths:np.array, mask:int=-1.0)->np.array:
        """Set fragments out-of-range to -1"""
        for i in range(array.shape[0]):
            array[i, lengths[i] - 1 :, :, :, :] = mask
        return array
    
    @staticmethod
    def reshape_dims(array: np.array)->np.array:
        """Reshape intensities"""
        MAX_SEQUENCE = 30
        ION_TYPES = ["y", "b"]
        MAX_FRAG_CHARGE = 3

        n, dims = array.shape
        assert dims == 174
        nlosses = 1
        return array.reshape(
            [array.shape[0], MAX_SEQUENCE - 1, len(ION_TYPES), nlosses, MAX_FRAG_CHARGE]
        )
    
    @staticmethod
    def mask_outofcharge(array: np.array, charges: np.array, mask: int=-1.0)->np.array:
        """Set entries that cannot exist to -1"""
        for i in range(array.shape[0]):
            if charges[i] < 3:
                array[i, :, :, :, charges[i] :] = mask
        return array
    
    @staticmethod
    def reshape_flat(array: np.array)->np.array:
        """Reshape intensities"""
        s = array.shape
        flat_dim = [s[0], functools.reduce(lambda x, y: x * y, s[1:], 1)]
        return array.reshape(flat_dim)

    @staticmethod
    def masked_spectral_distance(true:np.array, pred:np.array, epsilon:float = 1e-7):
        """Compute spectral angle"""
        pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
        true_masked = ((true + 1) * true) / (true + 1 + epsilon)
        

        pred_norm = normalize(pred_masked)
        true_norm = normalize(true_masked)
        product = np.sum(pred_norm * true_norm, axis=1)
        arccos = np.arccos(product)
        spectral_distance = 2 * arccos / np.pi
        return spectral_distance
    
    def getIntensitiesAndSpectralAngle(self, prediction: np.array, target: np.array, charge: np.array, sequence: np.array)->Tuple[np.array, np.array]:
        """Clean intensities and compute spectral angle"""
        sequence_lengths = [np.count_nonzero(s) for s in sequence]

        intensities = np.asarray(prediction)
        intensities_raw = np.asarray(target)
        charge = np.array(charge)
        charges = list(charge.argmax(axis=1) + 1)

        intensities[intensities < 0] = 0
        intensities = self.normalize_base_peak(intensities)
        intensities = self.reshape_dims(intensities)
        intensities = self.mask_outofrange(intensities, sequence_lengths)
        intensities = self.mask_outofcharge(intensities, charges)
        intensities = self.reshape_flat(intensities)

        spectral_angle = 1 - self.masked_spectral_distance(intensities_raw, intensities)
        spectral_angle = np.nan_to_num(spectral_angle)
        return spectral_angle, intensities


class PrepareTapeData(cleanTapeOutput, BatchLoader):
    """Prepate Tape HDF5-data by using Tape output and add meta-data from Prosit HDF5"""
    def __init__(self, prosit_hdf5_data: HDF5Matrix, tape_result: dict)->None:
        cleanTapeOutput.__init__(self)
        BatchLoader.__init__(self)
        
        self.prosit_hdf5_data = prosit_hdf5_data
        self.tape_result = tape_result
        
        self.prosit_keys_keep = list(set(prosit_hdf5_data.keys()) - set(['intensities_pred', 'spectral_angle']))
        self._keys = list(prosit_hdf5_data.keys())
        
        self._n_data_points, self.batch_size = len(tape_result), 100_000
        self.tape_data_hdf5 = {k:list() for k in self._keys}
        
    def getTapeIntensities(self, tape_batch:dict)->Tuple[List[np.array], List[np.array]]:
        """Get Tape intensities from batch"""
        tape_pred_intensities = [tape_batch[i]["prediction"] for i in range(len(tape_batch))]
        tape_target_intensities = [tape_batch[i]["target"] for i in range(len(tape_batch))]
        return tape_pred_intensities, tape_target_intensities
    
    def addTapeHDF5Data(self, prosit_batch: dict, cleanInt:np.array, sa:np.array)->None:
        """Add batch data"""
        for k in self.prosit_keys_keep:
            self.tape_data_hdf5[k].append(prosit_batch[k])
        self.tape_data_hdf5['intensities_pred'].append(cleanInt)
        self.tape_data_hdf5['spectral_angle'].append(sa)
    
    def concatenateTapeHDF5Data(self)->None:
        """Concatenate all batch-metadata to one final dataset"""
        tmp_data = dict()
        for k in self._keys:
            tmp_data[k] = np.concatenate(self.tape_data_hdf5[k])
        self.tape_data_hdf5 = tmp_data
        
             
    def createTapeHDF5Dict(self)->dict:
        """Create hdf5 data dict"""
        tape_data_hdf5 = {k:list() for k in self._keys}
        for c, (start, end) in enumerate(self.getBatchIxs(self._n_data_points, self.batch_size)):
            prosit_batch = {k: self.prosit_hdf5_data[k][start:end] for k in self._keys}
            

            tape_pred_intensities = self.tape_result[start:end]
            sa, cleanInt = self.getIntensitiesAndSpectralAngle(tape_pred_intensities, 
                                                               prosit_batch["intensities_raw"], 
                                                               prosit_batch["precursor_charge_onehot"], 
                                                               prosit_batch["sequence_integer"]
                                                              )
            self.addTapeHDF5Data(prosit_batch, cleanInt, sa)
        self.concatenateTapeHDF5Data()
        print("Tape Median Spectrum Angle {}".format(np.median(self.tape_data_hdf5['spectral_angle'])))
        return self.tape_data_hdf5
