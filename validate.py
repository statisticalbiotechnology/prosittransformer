import click
from prosittransformer.fileConverters import Tape2Prosit
from tqdm import tqdm
from prosittransformer.validation import checkAllClose, CompareSA
from prosittransformer.utils import hdf5Loader
from pathlib import Path
from prosittransformer.utils import PathHandler

@click.command()
@click.option('--tf_hdf5', type=click.Path(), help="Path to tf result.")
@click.option('--torch_hdf5', type=click.Path(), help="Path to torch result.")
def cli(tf_hdf5: Path, torch_hdf5: Path)->None:
    """Compare pytorch predictions with tensorflow predictions"""
    assert PathHandler.isFile(tf_hdf5), f"{tf_hdf5} don't exist!"
    assert PathHandler.isFile(torch_hdf5), f"{torch_hdf5} don't exist!"
    
    tf_hdf5 = hdf5Loader.from_hdf5(tf_hdf5)
    torch_hdf5 = hdf5Loader.from_hdf5(torch_hdf5)

    checkAllClose(torch_hdf5['intensities_pred'], tf_hdf5['intensities_pred'])
  
    CompareSA(torch_hdf5['spectral_angle'], tf_hdf5['spectral_angle'])
    