#!/bin/sh
for d in /sdd/berzelius/final_results/* ; do
    echo "$d"
    Rscript report.R \
    --val_file "$d/torchResult.hdf5" \
    --ho_file /sdd/PrositToTapeDataConverter/hdf5/hcd/HDF5/prediction_hcd_ho.hdf5 \
    --out_dir $d
done