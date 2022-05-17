#!/bin/sh
for d in /sdd/berzelius/final_results/* ; do
    #echo "$d"
    predictTorch --model "$d" --lmdb /sdd/PrositToTapeDataConverter/LMDB/ --split test --prosit_hdf5_path /sdd/PrositToTapeDataConverter/hdf5/hcd/HDF5/prediction_hcd_ho.hdf5 --out_dir "$d"
done