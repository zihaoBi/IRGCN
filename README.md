# Interaction-Aware Riemannian Graph Convolutional Network for Skeleton-Based Two-Person Interaction Recognition

## 1 Prerequisites

Our code is based on the 2P-GCN architecture (https://github.com/mgiant/2P-GCN).

### 1.1 Libraries

This code is based on [Python3](https://www.anaconda.com/) (anaconda, >= 3.8) and [PyTorch](http://pytorch.org/) (>= 2.3.0).

Other Python libraries are presented in the **'scripts/requirements.txt'**, which can be installed by

```
pip install -r scripts/requirements.txt
```

### 1.2 Experimental Dataset

**NTU RGB+D 60 & 120** datasets can be downloaded from [here](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp). There are 302 samples of **NTU RGB+D 60** and 532 samples of **NTU RGB+D 120** need to be ignored, which are shown in the **'src/reader/ntu_ignore.txt'**.

For **SBU** dataset, please refer to [SBU-Kinect-Interaction dataset v2.0](http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt). We use the clean version in our experiments.

## 2 Parameters

Before training and evaluating, there are some parameters should be noticed.

* (1) **'--config'** or **'-c'**: The path to configuration file. You must use this parameter in the command line or the program will output an error.
* (2) **'--work_dir'** or **'-w'**: The path to workdir, for saving checkpoints and other running files. Default is **'./workdir'**.
* (3) **'--pretrained_path'** or **'-pp'**: The path to the pretrained models. **pretrained_path = None** means using randomly initial model. Default is **None**.
* (4) **'--resume'** or **'-r'**: Resume from the recent checkpoint (**'<--work_dir>/checkpoint.pth.tar'**).
* (5) **'--evaluate'** or **'-e'**: Only evaluate models. You can choose the evaluating model according to the instructions.
* (6) **'--dataset'** or **'-d'**: Choose the dataset. (Choice: **[ntu, ntu120, ntu-mutual, ntu120-mutual, sbu]**)

Parameters can be updated by modifying the corresponding config file in the **'configs'** folder or using command line to send parameters to the model, and the parameter priority is **command line > yaml config > default value**.

## 3 Running

### 3.1 Modify Configs

Firstly, you should modify the **'path'** parameters in all config files of the **'configs'** folder.

A python file **'scripts/modify_configs.py'** will help you to do this. You need only to change three parameters in this file to your path to NTU datasets.

```
python scripts/modify_configs.py --root_folder <path/to/save/numpy/data> --ntu60_path <path/to/ntu60/dataset> --ntu120_path <path/to/ntu120/dataset>
```

All the commands above are optional.

### 3.2 Generate Datasets

After modifing the path to datasets, please generate numpy datasets before trainning (only the first time to use this benchmark). It may takes a couple of minues.

```
python main.py -c <path-to-config> -gd
```

where `<path-to-config>` is the (absolute or relative) path  to the config file, e.g., `configs/ntu_mutual/xview.yaml`.

For NTU-RGB+D dataset, if the `dataset` argument specifies `ntu_mutual`, the program will generate **ntu_mutual** dataset which only contains mutual actions; otherwise the whole NTU dataset will be generated.

### 3.3 Train

You can simply train the model by

```
python main.py -c <path-to-config>
```

For example

```
python main.py -c configs/ntu_mutual/xview.yaml
```

### 3.4 Evaluate

Before evaluating, you should ensure that the trained model corresponding the config is already existed in the **<--pretrained_path>** or **'<--work_dir>'** folder. Then run

```
python main.py -c <path-to-config> -e
```
