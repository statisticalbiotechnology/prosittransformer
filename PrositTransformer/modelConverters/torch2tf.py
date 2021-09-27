from .onnx2tf import onnx2tf
from .torch2onnx import torch2onnx
import tempfile

def torch2tf(torch_model: str, lmdb: str, tf_model: str, data_type: str)->None:
    """Convert torch model to tf"""
    temp_dir = tempfile.TemporaryDirectory()
    onnx_model = temp_dir.name + "/tape.onnx"
    torch2onnx(torch_model, lmdb, data_type, onnx_model)
    onnx2tf(tf_model, onnx_model)
    temp_dir.cleanup()