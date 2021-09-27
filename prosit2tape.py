import click
from PrositTransformer.fileConverters.Prosit2Tape import Prosit2Tape
from PrositTransformer.constants import splits
from pathlib import Path
from PrositTransformer.constants import splits

@click.command()
@click.option('--prosit_hdf5_path', type=click.Path(), help="Path to prosit hdf5 path.")
@click.option('--out_dir', type=click.Path(), help="Path to tape result in prosit hdf5 format.")
@click.option('--split', type=click.Path(), help="Select data split: train, test, or valid")
def cli(prosit_hdf5_path: Path, out_dir: Path, split: str)->None:
    """Convert prosit hdf5 file into tape lmdb file"""

    assert split in splits, f"{split} not valid. Needs to be any of {splits}"
    DC = Prosit2Tape.convertFromPath(split, prosit_hdf5_path, out_dir)
    DC.convert()