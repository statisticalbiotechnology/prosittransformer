import click
from tape import ProteinBertForValuePredictionFragmentationProsit
from tape.datasets import PrositFragmentationDataset
import numpy as np
import torch
from prosittransformer.DataHandler import getTorchDataLoader
from tqdm import tqdm
from pathlib import Path
from prosittransformer.constants import splits
from prosittransformer.fileConverters.Tape2Prosit import Tape2Prosit
import tempfile
from prosittransformer.utils import PathHandler
from tape import TAPETokenizer
import pandas as pd
from prosittransformer.DataHandler import pad_sequences

from tqdm import tqdm

from prosittransformer.prositInputHandler import PrositInputDataset
from torch.utils.data import DataLoader
from prosittransformer.prositUtils.tensorize import csv
from prosittransformer.prositUtils import sanitize
from prosittransformer.prositUtils.converters import generic
import tempfile
import pickle as pkl

@click.command()
@click.option('--model', type=click.Path(), help="Path to tape torch model.")
@click.option('--prosit_input', type=click.Path(), help="Path to tape torch model.")
@click.option('--irt_path', type=click.Path(), help="Path to tape torch model.")
@click.option('--out_dir', type=click.Path(), help="Path to tape torch model.")
@click.option('--batch_size', default=64, help="Batch size during eval")
def cli(model: Path, prosit_input: Path, irt_path:Path, out_dir: Path, batch_size:int):
    """Make prediction with tape Torch model and save result into prosit-hdf5"""
    assert PathHandler.isDir(model), f"{model} don't exist!"
    assert PathHandler.isDir(out_dir), f"{out_dir} don't exist!"
    assert PathHandler.isFile(prosit_input), f"{prosit_input} don't exist!"
    assert PathHandler.isFile(irt_path), f"{irt_path} don't exist!"

    df = pd.read_csv(prosit_input, sep=",")
    D = pkl.load(open(irt_path, "rb"))
    iRT = D["iRT"]
    
    dataset = PrositInputDataset(prosit_input)
    loader = DataLoader(dataset, num_workers=6,
                    collate_fn=dataset.collate_fn,
                    batch_size=batch_size)
    print("Fix masses")
    x = csv(df)
    print("Predict spectra")
    

    pytorch_model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(model)

    if torch.cuda.is_available():
        use_gpu = True
        pytorch_model = pytorch_model.to(torch.device('cuda:0'))
    else:
        use_gpu = False

    predictions = list()
    for batch in tqdm(loader):  
        if use_gpu:
            batch = {name: tensor.cuda(device=torch.device('cuda:0'), non_blocking=True)
                     for name, tensor in batch.items()}
        predictions.append(pytorch_model(**batch)[0].cpu().detach().numpy())
    preds = np.concatenate(predictions)

    x["intensities_pred"] = preds
    x["iRT"] = iRT

    data = sanitize.prediction(x)

    tmp_f = tempfile.NamedTemporaryFile(delete=False)
    c = generic.Converter(data, tmp_f.name)
    c.convert()
    I = pd.read_csv(tmp_f.name, ",")
    I.to_csv(f"{out_dir}/prosit_output.csv", sep=",", index=False)


    #np.save(f"{out_dir}/predicted_spectra.npy", preds)