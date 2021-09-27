# Installation
```console
prosit@transformer:~$ pip install -e .
```
Everything that is required will be install except for Nvidia Apex. Install it here: https://github.com/NVIDIA/apex\
However, only training requires Nvidia Apex
# Data
You can get the data in two ways.
### Prosit HDF5 to Tape LMDB
Download HDF5-files
```console
prosit@transformer:~$ bash downloadProsithdf5.sh
```
Then convert files to LMDB
```console
prosit@transformer:~$ prosit2tape \
--prosit_hdf5_path /path/to/file.hdf5 \
--out_dir /path/to/data \
--split test
```
The conversion takes a long time (~20h for training data)
### Download LMDB data
#TODO: Fix a download file for LMDB-data

# Training
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

We have an example config file at config/prositTransformer.json
# Model conversion
#TODO: make it possible to download models

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
#TODO: Make it possible to download our final prediction file

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