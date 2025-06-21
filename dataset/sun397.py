import torch
from datasets import load_dataset_builder, load_dataset
from torch.utils.data import Dataset, DataLoader

class HFSun397Dataset(Dataset):
    def __init__(self, hf_split, transforms=None):
        self.hf_split = hf_split
        self.transforms = transforms

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item['image'].convert("RGB")
        label = item['label']
        if self.transforms:
            image = self.transforms(image)
        return image, label

def _load_and_get_splits(config):
    full_dataset = load_dataset("tanganke/sun397", split="train", cache_dir=config.get('hf_cache_dir'))

    main_split_dict = full_dataset.train_test_split(train_size=0.8, seed=42)
    
    final_test_split_dict = main_split_dict['test'].train_test_split(test_size=0.5, seed=42)

    return {
        'train': main_split_dict['train'],
        'val': final_test_split_dict['train'],
        'test': final_test_split_dict['test']
    }


def prepare_train_loaders(config):
    splits = _load_and_get_splits(config)
    train_dataset = HFSun397Dataset(hf_split=splits['train'], transforms=config['train_preprocess'])

    loaders = {
        'full': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    }
    return loaders

def prepare_test_loaders(config):
    splits = _load_and_get_splits(config)
    
    val_dataset = HFSun397Dataset(hf_split=splits['val'], transforms=config['eval_preprocess'])
    test_dataset = HFSun397Dataset(hf_split=splits['test'], transforms=config['eval_preprocess'])

    loaders = {
        'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']),
        'test': DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']),
    }

    builder = load_dataset_builder("tanganke/sun397")
    raw_classnames = builder.info.features['label'].names
    final_classnames = [name[2:].replace('_', ' ') for name in raw_classnames]
    loaders['class_names'] = final_classnames
    
    return loaders