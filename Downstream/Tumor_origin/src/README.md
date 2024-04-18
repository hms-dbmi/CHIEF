
### Dataset
We have provided the data of publicaly available cases (TCGA) in our dataset to seamlessly run and validate the code.


### Training
``` shell
CUDA_VISIBLE_DEVICES=2 python3 train_valid_test.py --classification_type='tumor_origin' --exec_mode='train' --exp_name='tcga_only_7_1_2'
```
### Evaluation 
``` shell

CUDA_VISIBLE_DEVICES=0 python3 train_valid_test.py --classification_type='tumor_origin' --exec_mode='eval' --exp_name='tcga_only_7_1_2' --split_name='test' 
```
