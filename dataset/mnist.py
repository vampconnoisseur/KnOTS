import os
import torch
import torchvision.datasets as datasets
from utils import create_heldout_split


ROOT = "" 

class MNIST:
    def __init__(self,
                 is_train,
                 preprocess,
                 location=ROOT,
                 batch_size=128,
                 num_workers=16):
        if is_train:
            self.train_dataset = datasets.MNIST(
                root=location,
                download=True,
                train=True,
                transform=preprocess
            )

            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        
        else:
            self.test_dataset = datasets.MNIST(
                root=location,
                download=True,
                train=False,
                transform=preprocess
            )

            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def prepare_train_loaders(config):
    dataset_class = MNIST(
        is_train=True,
        preprocess=config['train_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader
    }
    return loaders

def prepare_test_loaders(config):
    dataset_class = MNIST(
        is_train=False,
        preprocess=config['eval_preprocess'],
        location=config.get('location', ROOT),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {
        'test': dataset_class.test_loader
    }
    
    if config.get('val_fraction', 0) > 0.:
        print('splitting mnist for validation')
        val_subset, test_subset = create_heldout_split(loaders['test'].dataset, fraction=config['val_fraction'])
        
        loaders['val'] = torch.utils.data.DataLoader(
            val_subset,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['test'] = torch.utils.data.DataLoader(
            test_subset,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        
    loaders['class_names'] = dataset_class.classnames
    return loaders