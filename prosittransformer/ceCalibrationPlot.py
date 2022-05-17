from torch.utils.data import Dataset
from tape.tokenizers import TAPETokenizer
from typing import List, Tuple, Any, Dict
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tape.utils._sampler import BucketBatchSampler
from tape.datasets import pad_sequences
from tape import ProteinBertForValuePredictionFragmentationProsit
from prosittransformer.utils import cleanTapeOutput
from tqdm import tqdm
import pickle
import multiprocessing
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from tape.datasets import PrositFragmentationDataset
import pandas as pd

class SelectCEData:
    """Select datapoints for certain CE's"""
    def __init__(self, lmdb : str, selected_ce : List[float] = [0.2, 0.25, 0.3, 0.35, 0.4]):
        self.PrositData = PrositFragmentationDataset(lmdb, "test")
        self._ceDataDict = {ce : [] for ce in selected_ce }
    @property
    def ceDataDict(self):
        return self._ceDataDict
    def getCEdata(self)->dict:
        """Get data from LMDB file"""
        #Loop each element in dataset
        for i in tqdm(range(len(self.PrositData))):
            for k in self._ceDataDict.keys():
                ce = np.round(self.PrositData[i][3], 2)
                if ce == np.array(k, dtype=np.float32):
                    self._ceDataDict[k].append(self.PrositData[i])
        return self._ceDataDict

class PrositFragmentationCEDataset(Dataset):
    """Dataset that set collision energy"""
    def __init__(self,
                 data: dict,
                 ce: float):

        tokenizer = TAPETokenizer(vocab="iupac")
        self.tokenizer = tokenizer
        self.data = data
        self.ce = ce
        self.keys = [
                     'intensities_raw',
                     'collision_energy_aligned_normed',
                     'precursor_charge_onehot'
                     ]
                     
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        #Set CE to a fixed value
        return self.data[index][:3] + tuple([np.array(self.ce, dtype=np.float32)]) + self.data[index][4:]

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, intensities_raw_true_value, collision_energy, charge = tuple(zip(*batch))

        collision_energy = np.stack(collision_energy)
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        intensities_raw_true_value = torch.FloatTensor(intensities_raw_true_value)  # type: ignore

        collision_energy_tensor = torch.FloatTensor(collision_energy)
        charge_tensor = torch.FloatTensor(charge)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': intensities_raw_true_value,
                'collision_energy': collision_energy_tensor,
                'charge': charge_tensor}

class CreateDataLoader:
    """Use BucketBatchSampler here since order doesnt matter in this case=>faster predictions"""
    @staticmethod
    def getDataLoader(dataset: PrositFragmentationDataset)->DataLoader:
        """get ce data loader"""
        sampler = RandomSampler(dataset)
        batch_sampler = BucketBatchSampler(sampler, 64, False, lambda x: len(x[0]), dataset)
        loader = DataLoader(
                dataset,
                num_workers=multiprocessing.cpu_count() - 2,
                collate_fn=dataset.collate_fn,  # type: ignore
                batch_sampler=batch_sampler)
        return loader
    
class PytorchModel:
    """Get model and load it on GPU"""
    @staticmethod
    def getModel(path : str)->ProteinBertForValuePredictionFragmentationProsit:
        """Get pytorch model loaded on GPU"""
        pytorch_model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(path)
        pytorch_model = pytorch_model.to(torch.device('cuda:0'))
        return pytorch_model

class CeCalibation(SelectCEData, cleanTapeOutput, CreateDataLoader, PytorchModel):
    """Generate Ce Calibration plot"""
    def __init__(self, lmdb: str, pytorch_model: str, out_dir: str, ce_range : np.array = np.linspace(0.10,0.5,41)):
        cleanTapeOutput.__init__(self)
        CreateDataLoader.__init__(self)
        PytorchModel.__init__(self)
        SelectCEData.__init__(self, lmdb)
        PytorchModel.__init__(self)
        self.model = self.getModel(pytorch_model)
        self.out_dir = out_dir
        self.ce_range = ce_range
        
    def _getSa(self, loader: DataLoader)->float:
        """Predict spectrum and get SA"""
        sa_list = list()
        for batch in loader:  
            batch = {name: tensor.cuda(device=torch.device('cuda:0'), non_blocking=True)
                         for name, tensor in batch.items()}
            targets = batch["targets"].cpu().detach().numpy()
            charge = batch["charge"].cpu().detach().numpy()
            sequence = batch["input_ids"].cpu().detach().numpy()
            predictions = self.model(**batch)[1].cpu().detach().numpy()
            sa, _ = self.getIntensitiesAndSpectralAngle(predictions, targets, charge, sequence, True)
            sa_list.append(sa)
        return np.median(np.concatenate(sa_list))
    
    def _getCeCalibSeries(self, data: dict, range_x: np.array)->List[float]:
        """Get SA for different calibration series"""
        sa_list = list()
        for i in tqdm(range_x):
            dataset = PrositFragmentationCEDataset(data, round(i,2))
            loader = self.getDataLoader(dataset)
            sa_list.append(self._getSa(loader))
        return sa_list
    
    def _makeFig(self, df: pd.DataFrame)->None:
        """Create figure"""
        fs=16 + 5
        plt.figure(figsize=(16, 10))
        ax = seaborn.lineplot(x="ce", y="sa", marker="o", hue="CE", data = df, palette=["C0", "C1", "C2","C3", "C4"])
        legend = ax.legend(handles=ax.legend_.legendHandles,
                           prop={"size":fs})
        plt.xlabel("Collision Energy", fontsize=fs)
        plt.ylabel("Median Spectral Angle", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.plot([0.2, 0.2], [0, 1], color="C0")
        plt.plot([0.25, 0.25], [0, 1], color="C1")
        plt.plot([0.3, 0.3], [0, 1], color="C2")
        plt.plot([0.35, 0.35], [0, 1], color="C3")
        plt.plot([0.4, 0.4], [0, 1], color="C4")
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}CeCalibation.png")
    
    def _getCeCalibationDF(self)->pd.DataFrame:
        """Create calibration dataset series"""
        print("Collect CE data")
        ceDataDict = self.getCEdata()
        data_points = list()
        print("Start getting spectral angle for all CE series")
        for ce in list(ceDataDict.keys()):
            print(f"Collecting SA for {ce} ce")
            CE_DATA = ceDataDict[ce]
            if len(CE_DATA) == 0:
                print(f"no data for {ce}. Skip to next ce.")
                continue
                
            sa_list = self._getCeCalibSeries(CE_DATA, self.ce_range)
            for s, r in zip(sa_list, self.ce_range):
                data_points.append([s,r, ce])    
        df = pd.DataFrame(data_points, columns=["sa", "ce", "CE"])
        return df
    
    def CeCalibarationReport(self)->None:
        """Get Ce calibration plot"""
        df = self._getCeCalibationDF()
        df.to_csv(f"{self.out_dir}CeCalibation.csv")
        self._makeFig(df)

        
    