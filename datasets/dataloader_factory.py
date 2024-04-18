import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

def create_dataloader(cfg):

    return create_bag_dataloader( cfg)


def create_bag_dataloader( cfg):
    df = pd.read_csv(cfg.Data.external_dir)

    df.rename(columns={'slide_id': 'case_id'}, inplace=True)
    df.rename(columns={'image_id': 'case_id'}, inplace=True)

    from datasets.BagDataset import BagDataset
    dataset = BagDataset(df, **cfg.Data)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                num_workers=1)

    return dataloader
