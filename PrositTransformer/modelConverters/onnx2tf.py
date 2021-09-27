import onnx
from onnx_tf.backend import prepare

def onnx2tf(tf_model: str, onnx_model: str):
    """Convert onnx model to tf"""
    model = onnx.load(onnx_model)
    tf_rep = prepare(model, device='GPU') 
    if not tf_model.endswith("/"):
        tf_model += "/"
    tf_rep.export_graph(f'{tf_model}/model.pb')