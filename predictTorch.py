import click
from tape import ProteinBertForValuePredictionFragmentationProsit
from tape.datasets import PrositFragmentationDataset
import numpy as np
import torch
from PrositTransformer.DataHandler import getTorchDataLoader
from tqdm import tqdm
from pathlib import Path
from PrositTransformer.constants import splits
from PrositTransformer.fileConverters.Tape2Prosit import Tape2Prosit
import tempfile

@click.command()
@click.option('--model', type=click.Path(), help="Path to tape torch model.")
@click.option('--lmdb', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--out_dir', type=click.Path(), help="Path to save predictions.")
@click.option('--out_file', default="torchResult.hdf5", type=click.Path(), help="File name for hdf5-file.")
@click.option('--split', default="test", help="Data split.")
@click.option('--prosit_hdf5_path', help="Path to prosit hdf5-file.")
@click.option('--batch_size', default=64, help="Batch size during eval")
def cli(model: Path, lmdb: Path, out_dir: Path, split: str, prosit_hdf5_path:Path, out_file:str, batch_size:int):
    """Make prediction with tape Torch model and save result into prosit-hdf5"""

    assert split in splits, f"{split} not valid. Needs to be any of {splits}"
    pytorch_model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(model)
    if torch.cuda.is_available():
        use_gpu = True
        pytorch_model = pytorch_model.to(torch.device('cuda:0'))
    else:
        use_gpu = False
   
    loader = getTorchDataLoader(lmdb, split, batch_size = batch_size)
    predictions = list()
    for batch in tqdm(loader):  
        if use_gpu:
            batch = {name: tensor.cuda(device=torch.device('cuda:0'), non_blocking=True)
                     for name, tensor in batch.items()}
        predictions.append(pytorch_model(**batch)[1].cpu().detach().numpy())
    predictions = np.concatenate(predictions)

    if not out_dir.endswith("/"):
        out_dir += "/"
    
    temp_dir = tempfile.TemporaryDirectory()
    tmp_file = f"{temp_dir.name}/preds.npy"
    np.save(tmp_file, predictions)
    CONVERT = Tape2Prosit.fromPath(prosit_hdf5_path, tmp_file)
    CONVERT.convert(f"{out_dir}{out_file}")
    temp_dir.cleanup()
        


        