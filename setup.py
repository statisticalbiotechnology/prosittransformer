from setuptools import setup

setup(
    name="torchToTF_tape",
    version="1.0",
    py_modules=['convert'],
    install_requires=[
        'Click',
        'tensorflow-gpu==2.3.0',
        'torch==1.8.1',
        'tqdm==4.62.3',
        'onnx==1.9.0',
        'onnx-tf==1.8.0',
        'scikit-learn',
        'scipy',
        'tape_proteins @ https://github.com/statisticalbiotechnology/tape/tarball/master'
    ],
    entry_points='''
    [console_scripts]
    torch2tf=torch2tf:cli
    predictTorch=predictTorch:cli
    predictTF=predictTF:cli
    validate=validate:cli
    prosit2tape=prosit2tape:cli
    '''
)
