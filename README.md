# CHIEF - Clinical Histopathology Imaging Evaluation Foundation Model


### A Generalizable Foundation Model for Quantitative Pathology Image Analyses 

by Wang. X, Zhao. J. et al.

#### ABSTRACT 
*Histopathology image evaluation is indispensable for cancer diagnoses and subtype classification. Standard
artificial intelligence (AI) methods for histopathology image analyses have focused on optimizing specialized models for each diagnostic task. Although such methods have achieved some success, they often have
limited generalizability to images generated by different digitization protocols or samples collected from different populations. To address this challenge, we devised the Clinical Histopathology Imaging Evaluation
Foundation Model (CHIEF), a general-purpose weakly supervised machine learning framework trained with
60,530 whole-slide images (WSIs) spanning 19 distinct cancer types. CHIEF leverages two complementary
pretraining methods to extract diverse pathology representations: unsupervised pretraining for tile-level feature identification and weakly supervised pretraining for whole-slide pattern recognition. Through pretraining
on 44 terabytes of high-resolution histopathology imaging datasets, CHIEF extracted pathology imaging representations useful for cancer detection, tumor origin identification, molecular profile characterization, and
survival outcome prediction. We successfully validated CHIEF using 24,765 whole-slide images from 38 independent slide sets collected from 24 hospitals and cohorts internationally. Overall, CHIEF outperformed the
state-of-the-art deep learning methods by up to 36.1% in these digital pathology evaluation tasks, showing its
ability to address domain shifts observed in samples from diverse populations and processed by different slide
preparation methods. CHIEF provides a generalizable foundation for efficient digital pathology evaluation for
cancer patients.*

![Github-Cover](https://github.com/hms-dbmi/CHIEF/assets/31292151/442391e2-3706-4337-ae9a-69c2cc24222e)

© This code is made available for non-commercial academic purposes. 

## Pre-requisites:
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce V100 x 32GB)
* Python (Python 3.8.10),torch==1.8.1+cu111,
torchvision==0.9.1+cu111, h5py==3.6.0, matplotlib==3.5.2, numpy==1.22.3, opencv-python==4.5.5.64, openslide-python==1.3.0, pandas==1.4.2, Pillow==10.0.0, scikit-image==0.21.0
scikit-learn==1.2.2,scikit-survival==0.21.0, scipy==1.8.0, tensorboardX==2.6.1, tensorboard==2.8.0.

### Installation Guide for Linux (using anaconda)
1. Installation anaconda(https://www.anaconda.com/distribution/)
```
2. sudo apt-get install openslide-tools
```
```
3. pip install requirements.txt
```
### Step 1 Data Preparation
1.Download data
* [TCGA](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D), [CPTAC](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D), [Panda](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data),
 [DiagSet](https://github.com/michalkoziarski/DiagSet), and so on.


2.Get tiles embeddings
```
python3 get_feature.py
```
You need to process the WSI into the following format.Here some [examples](https://github.com/hms-dbmi/CHIEF/tree/main/Tasks/survial/features)
```
DATA_DIR
├─patch_coord
│      slide_id_1.h5
│      ...
└─patch_feature
        slide_id_1.pt
        ...
```

```python
coords = h5py.File(coords_path, 'r')['coords'][:]
# coords is a array like:
# [[x1, y1], [x2, y2], ...]
```

The pt file in the `patch_feature`folder contains the features of each patch of the WSI, which can be read as

```python
features = torch.load(features_path, map_location=torch.device('cpu'))
# features is a tensor with dimension N*F, and if features are extracted using CTransPath, F is 768
```


### Step 2. preparing the data set split

You need to divide the dataset into a training set validation set and a test set, and store them in the following format

```
SPLIT_DIR
    test_set.csv
    train_set.csv
    val_set.csv
```

And, the format of the csv file is as follows

| slide_id   | label |
| ---------- | ----- |
| slide_id_1 | 0     |
| slide_id_2 | 1     |
| ...        | ...   |
## Train model

### Step 1. create a config file

We have prepared two config file templates (see ./configs/) for CHIEF and other baselines, like

```yaml
General:
    seed: 7
    work_dir: WORK_DIR
    fold_num: 4

Data:
    split_dir: SPLIT_DIR
    data_dir: DATA_DIR # for data with no overlap
    features_size: 768
    n_classes: 2

Model:
    network: 'CHIEF'

Train:
    mode: repeat
    lr: 3.0e-4
    reg: 1.0e-5
    CosineAnnealingLR:
        T_max: 30
        eta_min: 1.0e-7
    Early_stopping:
        patient: 10
        stop_epoch: 30
        type: max
    max_epochs: 100
    train_method: CHIEF
    val_method: CHIEF
    dataset: BagDataset

Test:
    test_set_name:
        data_dir: TEST_DATA_DIR
        csv_path: TEST_SET_CSV_PATH
```

In the config, the correspondence between the `Model.network`, `Train.training_method` and `Train.val_method` is as follows

| `Model.network` | `Train.training_method` | `Train.val_method` |
|-----------------| ----------------------- | ------------------ |
| CHIEF           | CHIEF                   | CHIEF              |

### Step 2. train model

Run the following command

```shell
python train.py --config_path [config path] --begin [begin index] --end [end index]
```

`--begin` and `--end` used to control repetitive experiments

When the training is finished, the code creates a directory with the same name as the config file in the working directory to store all the experimental data. Like this

```
Task_DIR
│  CONFIG_FILE.yaml
│  s_0_checkpoint.pt
│  s_1_checkpoint.pt
│  s_2_checkpoint.pt
│  s_3_checkpoint.pt
│  s_4_checkpoint.pt
│
└─results
    │
    └─test_set
            metrics.csv
            probs.csv
```



## Evaluation

Please add information about the test set in the config, like this

```yaml
Test:
    TEST_SET_NAME:
        data_dir: TEST_DATA_DIR
        csv_path: TEST_SET_CSV_PATH
```

And run the command

```shell
python eval.py --config_path [config path] --dataset_name [TEST_SET_NAME]
```

Test result will save at `WORK_DIR/CONFIG_FILE_NAME/results/TEST_SET_NAME`

## Reproducibility

Below we provide a quick example using a subset of cases for RCC survival task.

```shell
cd Tasks/survial

run inference.ipynb
```
## Reference

* [SupContrast: Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast)
* [CLAM](https://github.com/mahmoodlab/CLAM)


## Issues
- Please open new threads or address all questions to xiyue.wang.scu@gmail.com or Kun-Hsing_Yu@hms.harvard.edu

## License
CHIEF is made available under the GPLv3 License and is available for non-commercial academic purposes. 

