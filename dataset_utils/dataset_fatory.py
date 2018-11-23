from dataset_utils.lpr_dataset import lpr_dataset
import torch
def make_dataloader(cfg):
    if cfg.task == 'lpr':
        print("generate lpr dataset")
        dataset = lpr_dataset(cfg.lpr_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = cfg.batch_size, shuffle = True,
                                        )
    return dataloader

