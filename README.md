# Installation
```console
prosit@transformer:~$ pip install -e .
```
Everything that is required will be install except for Nvidia Apex. Install it here: https://github.com/NVIDIA/apex\
However, only training requires Nvidia Apex
# Downloads: Data & Models
### Tape training data
You can download the TAPE LMDB training data here:
https://figshare.com/account/home#/projects/123637

You can also convert Prosit HDF5-files to TAPE LMDB data by using this command:


Then convert files to LMDB
```console
prosit@transformer:~$ prosit2tape \
--prosit_hdf5_path /path/to/file.hdf5 \
--out_dir /path/to/data \
--split test
```

### Prosit HDF5 files
You will need at least the Hold out file for benchmarking Prosit Transformer model. You can download it here:
https://figshare.com/articles/dataset/ProteomeTools_non_tryptic_-\_Prosit_fragmentation_-_Data/12937092

It's the hcd files that's being used.

### Models
You can download the torch and tensorflow version of the prosit transformer here:
https://figshare.com/account/home#/projects/123637

# Training
## Warning: It's highly recommended to create a new environment and install tape at https://github.com/statisticalbiotechnology/tape and install Nvidia-apex. There will most likely be compatibility issues with Nvidia-apex and torch running "tape-train" in the same environment as prosittransformer.
```console
prosit@transformer:~$ tape-train \
    transformer \
    prosit_fragmentation \
  --batch_size BS \
  --learning_rate LR \
  --warmup_steps WS \
  --gradient_accumulation_steps GAS \
  --fp16 \
  --data_dir /path/to/data \
  --patience P \
  --num_train_epochs EPOCHS \
  --output_dir /path/to/result \
  --log_dir /path/to/log \
  --exp_name NAME \
  --model_config_file /path/to/config/file.json
```

We have an example config file at config/prosittransformer.json
# Model conversion

In order to convert the Torch model to TensorFlow use:
```console
prosit@transformer:~$ torch2tf \
--torch_model /path/to/torch_model \
--lmdb /path/to/tape/data \
--tf_model path/to/tf_model
```
# Prediction
To make predictions, you need Prosit HDF5-files to get correctly formatted result-files for further validation and downstream analysis.

Download HDF5-files
```console
prosit@transformer:~$ bash downloadProsithdf5.sh
```

To make prediction you can use either
```console
prosit@transformer:~$ predictTorch \
--model /path/to/torch_model \
--lmdb /path/to/lmdb_data \
--split test \
--out_dir /path/to/predict_result
--prosit_hdf5_path /path/to/prosit/file_test.hdf5
```

or

```console
prosit@transformer:~$ predictTF \
--model /path/to/TF_model \
--lmdb /path/to/lmdb_data \
--split test \
--out_dir /path/to/predict_result \
--prosit_hdf5_path /path/to/prosit/file_test.hdf5
```

# Validate models
To test if the converted TF model works pass the result files after running the predict functions mentioned above
```console
prosit@transformer:~$ validate \
--tf_hdf5 /path/to/predict_result/tfResult.hdf5 \
--torch_hdf5 /path/to/predict_result/torchResult.hdf5
```
# Generate Prosit report

To get a Prosit report you need to make a predicton with any of the functions mentioned above to get result.hdf5 file. You also need prosit hdf5-file which you can get by running 

Download HDF5-files
```console
prosit@transformer:~$ bash downloadProsithdf5.sh
```

To get Prosit report run:
```console
prosit@transformer:~/prositReport$ cd prositReport
prosit@transformer:~$ cd Rscript report.R \
--val_file /path/to/predict_result/tfResult.hdf5 \
--ho_file /path/to/prosit/file_test.hdf5 \
--out_dir /path/to/report
```

# Generate CE Calibration plot
```console
prosit@transformer:~$ ceCalibration \
--torch_model /path/to/model \
--lmdb /path/to/LMDB \
--out_dir /path/to/output
```