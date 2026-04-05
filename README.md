# Adapt via Bayesian Nonparametric Clustering (ABC)

## Overview

Adapt via Bayesian Nonparametric Clustering (ABC) is a novel framework designed for Source Free Domain Adaptation scenarios where unknown target classes are present.


## Dependencies & Setup
Please run in Python3.10+ environment. We recommend using a virtual environment to manage dependencies.

Install all dependencies:
```bash
pip install -r requirements.txt
```

Before running any experiments, compile the Cython extension:
```bash
python setup.py build_ext --inplace
```


## Training

**1. Download Datasets**

Download the following datasets from their official websites and unzip them into the `./data` folder with the structure below:
```
./data
├── office
│   ├── amazon
│   │   └── ...
│   ├── amazon.txt
│   └── ...
├── OfficeHome
│   ├── Art
│   │   └── ...
│   ├── Art.txt
│   └── ...
└── visda
    ├── train
    │   └── train.txt
    └── validation
        └── validation.txt
```

**2. Pretrain Source Models**

We use the source model pretraining procedure from [GLC](https://github.com/ispc-lab/GLC). Place the pretrained models in `pretrained_source/OPDA/` with the naming convention `{dataset}_{source}.pkl`. For example:
```
pretrained_source/OPDA/office_amazon.pkl
```

**3. Run Training**

Run open-partial domain adaptation on Office, OfficeHome, and VisDA:
```bash
bash train.sh
```
## License
This code is released under the MIT License.