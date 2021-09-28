import numpy as np
from tqdm import tqdm
from .fileConverters.Tape2Prosit import Tape2Prosit
from tensorflow.keras.utils import HDF5Matrix

def checkAllClose(X : HDF5Matrix, Y : HDF5Matrix)->None:
    """Check how close all datapoints on different tolerance levels"""
    rtolDict = {"1e-1" : list(), "1e-2" : list(), "1e-3" : list(), "1e-4" : list()}

    print("Start to compare Torch and TF model outputs")
    for x, y in tqdm(zip(X, Y), total=len(X)):        
        for tol in rtolDict.keys():
            rtolDict[tol].append(np.allclose(x, y, rtol=float(tol)))

    for tol, result in rtolDict.items():
        print(f"{sum(result)} out of {len(result)} passed tol {tol}")

def CompareSA(x_sa: HDF5Matrix, y_sa: HDF5Matrix)->None:
    """Compare sa from two models"""
    delta = list()
    for x,y in tqdm(zip(x_sa, y_sa), total=len(x_sa)):
        delta.append(np.abs(x - y))
    median_sa_error, mean_sa_error = np.median(delta), np.mean(delta)
    print(f"Median sa error: {median_sa_error}\nMean sa error: {mean_sa_error}")