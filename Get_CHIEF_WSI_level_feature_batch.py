import torch, torchvision
import torch.nn as nn
from models.CHIEF import CHIEF
from datasets.dataloader_factory import create_dataloader
from utils.utils import read_yaml
import argparse
from tqdm import tqdm
import os
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/get_wsi_level_feature_exsample.yaml')
parser.add_argument('--dataset_name', type=str, default='test_set')
args = parser.parse_args()

cfg = read_yaml(args.config_path)
result_dir = os.path.join(cfg.General.result_dir,'WSI_level_feature', args.dataset_name)

os.makedirs(result_dir, exist_ok=True)
model = CHIEF(size_arg="small", dropout=True, n_classes=2)
model=model.cuda()
td = torch.load(r'./model_weight/CHIEF_pretraining.pth')
model.load_state_dict(td, strict=True)
model.eval()

dataloader = create_dataloader(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with torch.no_grad():
    with tqdm(total=len(dataloader)) as bar:
        for idx, batch in enumerate(dataloader):
            x, tmp_z,id = batch['x'].to(device, dtype=torch.float32), \
                batch['z'].to(device, dtype=torch.long),batch['id']
            result = model(x, x_anatomic=tmp_z)
            wsi_feature_emb = result['WSI_feature']  ###[1,768]
            print(wsi_feature_emb.size())
            torch.save(wsi_feature_emb, os.path.join(result_dir,id+'.pt'))




