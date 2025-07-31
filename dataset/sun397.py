import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset

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

def prepare_train_loaders(config):
    cache_dir = config.get('hf_cache_dir') or config.get('cache_dir') or None
    hf_train_split = load_dataset("tanganke/sun397", split="train", cache_dir=cache_dir)
    
    train_dataset = HFSun397Dataset(
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
    cache_dir = config.get('hf_cache_dir') or config.get('cache_dir') or None
    hf_full_test_split = load_dataset("tanganke/sun397", split="test", cache_dir=cache_dir)

    loaders = {}
    if config.get('val_fraction', 0) > 0.:
        print('Splitting SUN397 for validation')
        test_dataset_wrapped = HFSun397Dataset(hf_full_test_split, transforms=config['eval_preprocess'])
        
        shuffled_idxs_path = config['shuffled_idxs']
        if not os.path.exists(shuffled_idxs_path):
            print(f"Shuffled index file not found at {shuffled_idxs_path}. Creating it now...")
            os.makedirs(os.path.dirname(shuffled_idxs_path), exist_ok=True)
            indices = torch.randperm(len(test_dataset_wrapped))
            torch.save(indices, shuffled_idxs_path)
            print("Saved new shuffled indices.")
            
        shuffled_idxs = torch.load(shuffled_idxs_path)
        shuffled_idxs = shuffled_idxs.tolist()
        num_valid = int(len(test_dataset_wrapped) * config['val_fraction'])
        valid_idxs, test_idxs = shuffled_idxs[:num_valid], shuffled_idxs[num_valid:]
        
        val_subset = Subset(test_dataset_wrapped, valid_idxs)
        test_subset = Subset(test_dataset_wrapped, test_idxs)
        
        loaders['val'] = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        loaders['test'] = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    else:
        test_dataset_wrapped = HFSun397Dataset(hf_full_test_split, transforms=config['eval_preprocess'])
        loaders['test'] = DataLoader(test_dataset_wrapped, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    loaders['class_names'] = hf_full_test_split.features['label'].names
    
    return loaders