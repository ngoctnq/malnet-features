# Graph-Based Android Malware Classification for MalNet-Tiny-Features

This repository contains all code that were used to evaluate our experiments in the paper.

## Dataset Folder Structure
The datasets should be placed in the `datasets` folder. The structure of the folder should be:
```
datasets/
├── [dataset_name]/
│   ├── raw/
│   │   ├── malnet-graph-tiny/
│   │   │   ├── [malware_family]/[malware_type]/
│   │   │   │   ├── 0A1B2C...[.edgelist|.pkl]
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── split_info_tiny/type/
│   │       ├── train.txt
│   │       ├── val.txt
│   │       └── test.txt
│   └── processed/
└── ...
```

The code with check for `processed` versions first, and if there isn't any, it will start processing the raw data located in `raw`.

### Raw data
The structure of the raw data folder closely follows the original dataset, which can be downloaded [here](http://malnet.cc.gatech.edu/graph-data/).
- For the original MalNet-Tiny :
    - `malnet-graph-tiny` contains `.edgelist` files that represents the FCG structure.
    - `split_info_tiny` should be downloaded from the [original source](https://github.com/safreita1/malnet-graph/tree/master/split_info_tiny).
- For our new distribution-shifted datasets:
    - `malnet-graph-tiny` contains the constructed `.pkl` files,
    - `split_info_tiny` should be copied from the `feature-construction` folder.

### Processed data

## Environment Setup
Follow the instructions in `Exphormers.md` to setup the environment.

## How To Run
Construct the configuration files, and run the following syntax:
```bash
python main.py --cfg <yaml_file>
```

All experiments in the main paper have their configurations provided in `./configs`. Consult `README.md` in that directory for an explanation of the directory legends.

## Credit
- This codebase is built on top of the original [Exphormer code](https://github.com/hamed1375/Exphormer).
- GTrans code are adapted from their [original implementation](https://github.com/ChandlerBang/GTrans).