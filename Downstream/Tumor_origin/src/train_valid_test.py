from __future__ import print_function

import os
import argparse
import json

# internal imports
from classification import Tumor_origin

parser = argparse.ArgumentParser(description='Model Training Script')

# general parameters
parser.add_argument('--gt_csv', type=str, default='../data/train_valid_test/ground_truth.csv')
parser.add_argument('--split_csv', type=str, default='../data/train_valid_test/split.csv')

parser.add_argument('--train_csv', type=str, default='./csv/train_tcga.csv')
parser.add_argument('--val_csv', type=str, default='./csv/val_tcga.csv')
parser.add_argument('--test_csv', type=str, default='./csv/test_tcga.csv')

parser.add_argument('--histology_feature_path', type=str, default='./feature/tcga')
parser.add_argument('--results_dir', default='../results/', help='results directory (default: ../results)')
parser.add_argument('--exp_name', type=str, default='exp_01')

# training related parameters
parser.add_argument('--classification_type', type=str, default='tumor_origin')
parser.add_argument('--exec_mode', type=str, choices=['train', 'eval', 'interpret'], default='train')
parser.add_argument('--split_name', type=str, choices=['test'], default='test')
parser.add_argument('--model_name', type=str, default='MM_MLP')
parser.add_argument('--hidden_neurons_mm', type=int, default=512, help='features length')
parser.add_argument('--dropout', type=float, default=0.5, help='enabel dropout (p=0.25)')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')


# experiment related parameters
parser.add_argument('--max_epochs', type=int, default=1500, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--minimum_epochs', type=int, default=150, help='maximum number of epochs befor early stopping (default: 50)')
parser.add_argument('--patience', type=int, default=100, help='maximum number of epochs to wait for loss decreas before early stopping (default: 20)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=2, help='end fold (default: -1, first fold)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--site_name', type=str, default='Metastatic Recurrence')  # 'Metastatic Recurrence'
parser.add_argument('--balance_met', action='store_true', default=False)  # 'Metastatic Recurrence'

args = parser.parse_args()
args.results_dir = os.path.join(args.results_dir, args.exp_name)
args.ckpt_path = os.path.join(args.results_dir, 'checkpoint.pt')
os.makedirs(args.results_dir, exist_ok=True)
with open(os.path.join(args.results_dir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

args.label_dict = {'Prostate': 0, 'Lung': 1, 'Endometrial': 2, 'Breast': 3, 'Head Neck': 4, 'Colorectal': 5,
                   'Thyroid': 6, 'Skin': 7, 'Esophagogastric': 8, 'Ovarian': 9, 'Glioma': 10, 'Bladder': 11,
                   'Adrenal': 12, 'Renal': 13, 'Germ Cell': 14, 'Pancreatobiliary': 15, 'Liver': 16, 'Cervix': 17}
class_names = [class_name for class_name in args.label_dict.keys()]
args.n_classes = len(args.label_dict)

if __name__ == "__main__":
    obj = Tumor_origin(args)
    if args.exec_mode == 'train':
        obj.train_valid()
    elif args.exec_mode == 'eval':
        obj.eval(split_name=args.split_name)
