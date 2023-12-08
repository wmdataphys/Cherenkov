import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import torch

def DIRC_collate(batch):
    opt_boxes = []
    conditions = []
    PIDs = []
    for opt_box,cond,PID in batch:
        opt_boxes.append(torch.tensor(opt_box))
        conditions.append(torch.tensor(cond))
        PIDs.append(torch.tensor(PID))

    return torch.stack(opt_boxes), torch.stack(conditions),torch.tensor(PIDs)

def CreateLoaders(dataset,config):
    if config['dataloader']['mode'] == 'random':
        print("Using a random split.")
        n_images = dataset.__len__()

        split = np.array(config['dataloader']['split'])
        if(np.sum(split)!=1):
            raise NameError('The sum of the train/val/test split should be equal to 1')
            return

        n_train = int(config['dataloader']['split'][0] * n_images)
        n_val = int(config['dataloader']['split'][1] * n_images)
        n_test = n_images - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val,n_test], generator=torch.Generator().manual_seed(config['seed']))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['dataloader']['mode'])
        print('      Train Val ratio:', config['dataloader']['split'])
        print('      Training:', len(train_dataset),' indexes...',train_dataset.indices[:3])
        print('      Validation:', len(val_dataset),' indexes...',val_dataset.indices[:3])
        print('      Test:', len(test_dataset),' indexes...',test_dataset.indices[:3])
        print('')

    else:
        print("Using predefined indices for training/testing/validation.")

        train_ids = list(np.load(r"C:\Users\James-PC\James\EIC\Cherenkov\Data\train_indices.npy"))
        val_ids = list(np.load(r"C:\Users\James-PC\James\EIC\Cherenkov\Data\val_indices.npy"))
        test_ids = list(np.load(r"C:\Users\James-PC\James\EIC\Cherenkov\Data\test_indices.npy"))

        train_dataset = Subset(dataset,train_ids)
        val_dataset = Subset(dataset,val_ids)
        test_dataset = Subset(dataset,test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', "Preset Indices")
        print('      Train Val ratio:', config['dataloader']['split'])
        print('      Training:', len(train_dataset),' indexes...',train_dataset.indices[:3])
        print('      Validation:', len(val_dataset),' indexes...',val_dataset.indices[:3])
        print('      Test:', len(test_dataset),' indexes...',test_dataset.indices[:3])
        print('')

    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,
                            collate_fn=DIRC_collate,
                            pin_memory=True)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,
                            collate_fn=DIRC_collate,
                            pin_memory=True)
    test_loader =  DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size'],
                            shuffle=False,
                            collate_fn=DIRC_collate,
                            pin_memory=True)

    return train_loader,val_loader,test_loader
