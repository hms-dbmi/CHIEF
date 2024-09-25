import torch, torchvision
import torch.nn as nn
from models.CHIEF import CHIEF


model = CHIEF(size_arg="small", dropout=True, n_classes=2)

td = torch.load(r'./model_weight/CHIEF_pretraining.pth')
model.load_state_dict(td, strict=True)
model.eval()

full_path = r'./Downstream/Tumor_origin/src/feature/tcga/TCGA-LN-A8I1-01Z-00-DX1.F2C4FBC3-1FFA-45E9-9483-C3F1B2B7EF2D.pt'

features = torch.load(full_path, map_location=torch.device('cpu'))
anatomical=13
with torch.no_grad():
    x,tmp_z = features,anatomical
    result = model(x, torch.tensor([tmp_z]))
    wsi_feature_emb = result['WSI_feature']  ###[1,768]
    print(wsi_feature_emb.size())

