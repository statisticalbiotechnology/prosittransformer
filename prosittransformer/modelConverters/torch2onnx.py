import torch
from tape import ProteinBertForValuePredictionFragmentationProsit
import numpy as np
from ..DataHandler import getTFDataLoader

def torch2onnx(torch_model:str, lmdb:str, data_type:str, onnx_model:str)->None:
    """Convert torch model to onnx"""
    pytorch_model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(torch_model)

    loader = getTFDataLoader(lmdb, data_type, batch_size=1, returnTorchTensor=True)
    batch = next(iter(loader))
    torch.onnx.export(pytorch_model, args=tuple(batch.values()), f=onnx_model, input_names=list(batch.keys()),
        output_names=["output1"], export_params=True,opset_version=11, do_constant_folding=True,
                 dynamic_axes={'input_ids': [0], #this means second axis is dynamic       
                                    'input_mask' : [0],
                                   'collision_energy' : [0],
                                   'charge' : [0]
                                    })