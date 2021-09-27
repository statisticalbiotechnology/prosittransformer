import click
from PrositTransformer.modelConverters import torch2tf
from pathlib import Path
from PrositTransformer.constants import splits

@click.command()
@click.option('--torch_model', type=click.Path(), help="Path to tape torch model.")
@click.option('--lmdb', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--tf_model', type=click.Path(), help="Path to LMBD data folder.")
@click.option('--split', default="test", help="Path to LMBD data folder.")
def cli(torch_model: Path, lmdb: Path, tf_model: Path, split: str):
    """Convert pytorch tape model to tensorflow"""

    assert split in splits, f"{split} not valid. Needs to be any of {splits}"
    torch2tf(torch_model, lmdb, tf_model, split)
    