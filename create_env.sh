conda create -y -n prtr python=3.7 pip
conda activate prtr
conda install -y -c conda-forge cudatoolkit
conda install -y Click tqdm==4.62.3
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install onnx==1.9.0 onnx-tf==1.8.0 tensorflow-gpu tensorflow-io
conda install -y scikit-learn scipy
conda install -c conda-forge ipdb
conda install -y -c bioconda pyteomics spectrum_utils
pip install -e . 
