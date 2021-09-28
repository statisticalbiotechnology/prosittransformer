import click
from prosittransformer.ceCalibrationPlot import CeCalibation
from prosittransformer.utils import PathHandler
from pathlib import Path

@click.command()
@click.option('--torch_model', type=click.Path(), help="Path to tape torch model.")
@click.option('--lmdb', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--out_dir', type=click.Path(), help="Path to our directory.")
def cli(torch_model: Path, lmdb: Path, out_dir: Path):
    """Convert pytorch tape model to tensorflow"""
    assert PathHandler.isDir(torch_model), f"{torch_model} don't exist!"
    assert PathHandler.isDir(lmdb), f"{lmdb} don't exist!"
    assert PathHandler.isDir(out_dir), f"{tf_model} don't exist!"
    if not out_dir.endswith("/"):
        out_dir += "/"    
    
    Calib = CeCalibation(lmdb, torch_model, out_dir)
    Calib.CeCalibarationReport()