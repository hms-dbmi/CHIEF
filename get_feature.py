
import os
import h5py
import openslide
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


from models.ctran import ctranspath

def eval_transforms(size=256):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    trnsfrms_val = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
         ]
    )
    return trnsfrms_val


class WSI_dataset(Dataset):

    def __init__(self, wsi_path, h5_path, size):

        dset = h5py.File(h5_path, 'r')['coords']
        self.patch_size = dset.attrs['patch_size']
        self.coords = dset[:]
        self.wsi = openslide.open_slide(wsi_path)
        self.roi_transforms = eval_transforms(size=size)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, 0, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.roi_transforms(img)
        return img

def featurize_single_wsi(model, wsi_path, patch_coord_dir, patch_feature_dir, size=224, batch_size=32):

    slide_id = '.'.join(os.path.basename(wsi_path).split('.')[:-1])
    h5_path = os.path.join(patch_coord_dir, slide_id + '.h5')
    pt_path = os.path.join(patch_feature_dir, slide_id + '.pt')


    dataset = WSI_dataset(wsi_path, h5_path, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    features = []
    for batch in dataloader:
        with torch.no_grad():
            features.append(model(batch))

    features = torch.cat(features, dim=0).cpu()
    torch.save(features, pt_path)


model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'./ctranspathv2.pth')
model.load_state_dict(td['model'], strict=True)

wsi_path=r'./wsi_images/'
patch_coord_dir=r'./coord_dir/'
patch_feature_dir=r'./patch_feature_dir'

featurize_single_wsi(model, wsi_path, patch_coord_dir, patch_feature_dir)