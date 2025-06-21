import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class HFEuroSATDataset(Dataset):
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

def prepare_train_loaders(config):
    hf_train_split = load_dataset("tanganke/eurosat", split="train", cache_dir=config.get('hf_cache_dir'))
    
    train_dataset = HFEuroSATDataset(
        hf_split=hf_train_split,
        transforms=config['train_preprocess']
    )

    loaders = {
        'full': DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
    }
    return loaders

def prepare_test_loaders(config):
    hf_full_test_split = load_dataset("tanganke/eurosat", split="test", cache_dir=config.get('hf_cache_dir'))
    split_dict = hf_full_test_split.train_test_split(test_size=0.8, seed=42)
    hf_val_split = split_dict['train']
    hf_test_split = split_dict['test']
    
    val_dataset = HFEuroSATDataset(hf_split=hf_val_split, transforms=config['eval_preprocess'])
    test_dataset = HFEuroSATDataset(hf_split=hf_test_split, transforms=config['eval_preprocess'])

    loaders = {
        'val': DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']),
        'test': DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']),
    }
    
    final_classnames = load_dataset("tanganke/eurosat", split="train").features['label'].names
    
    print("Final classnames loaded directly from dataset:", final_classnames)
    
    loaders['class_names'] = final_classnames
    
    return loaders