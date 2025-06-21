import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge",
    "chaparral", "church", "circular_farmland", "cloud", "commercial_area",
    "dense_residential", "desert", "forest", "freeway", "golf_course",
    "ground_track_field", "harbor", "industrial_area", "intersection", "island",
    "lake", "meadow", "medium_residential", "mobile_home_park", "mountain",
    "overpass", "palace", "parking_lot", "railway", "railway_station",
    "rectangular_farmland", "river", "roundabout", "runway", "sea_ice", "ship",
    "snowberg", "sparse_residential", "stadium", "storage_tank", "tennis_court",
    "terrace", "thermal_power_station", "wetland",
]

class HFDataset(Dataset):
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
    hf_train_split = load_dataset("tanganke/resisc45", split="train", cache_dir=config.get('hf_cache_dir'))
    
    train_dataset = HFDataset(
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
    hf_full_test_split = load_dataset("tanganke/resisc45", split="test", cache_dir=config.get('hf_cache_dir'))
    
    split_dict = hf_full_test_split.train_test_split(test_size=0.5, seed=42)
    hf_val_split = split_dict['train']
    hf_test_split = split_dict['test']

    val_dataset = HFDataset(
        hf_split=hf_val_split,
        transforms=config['eval_preprocess']
    )
    test_dataset = HFDataset(
        hf_split=hf_test_split,
        transforms=config['eval_preprocess']
    )

    loaders = {
        'val': DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        ),
    }

    loaders['class_names'] = [' '.join(c.split('_')) for c in CLASSES]
    
    return loaders