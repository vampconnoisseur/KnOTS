import os, torch, numpy as np
from PIL import Image
from torchvision import datasets
from torchvision.datasets.utils import download_url
from tqdm.auto import tqdm
from utils import create_heldout_split

def _convert_svhn_mat_to_imagefolder(root_dir):
    image_dir = os.path.join(root_dir, 'images')
    if os.path.exists(os.path.join(image_dir, 'train')):
        print("SVHN images directory already exists. Skipping pre-processing.")
        return image_dir
    print("Performing one-time pre-processing of SVHN .mat files...")
    os.makedirs(image_dir, exist_ok=True)
    import scipy.io as sio
    for split in ['train', 'test']:
        mat_path = os.path.join(root_dir, f"{split}_32x32.mat")
        if not os.path.exists(mat_path):
            url = f"http://ufldl.stanford.edu/housenumbers/{split}_32x32.mat"
            download_url(url, root=root_dir, filename=os.path.basename(mat_path))
        loaded_mat = sio.loadmat(mat_path)
        data, labels = loaded_mat['X'], loaded_mat['y'].astype(np.int64).squeeze()
        np.place(labels, labels == 10, 0)
        split_path = os.path.join(image_dir, split)
        os.makedirs(split_path, exist_ok=True)
        print(f"Converting {split} split to image files...")
        for i in tqdm(range(data.shape[3])):
            img = Image.fromarray(np.transpose(data[:, :, :, i], (1, 0, 2)))
            label_dir = os.path.join(split_path, str(labels[i]))
            os.makedirs(label_dir, exist_ok=True)
            img.save(os.path.join(label_dir, f"{i}.png"))
    print("SVHN pre-processing complete.")
    return image_dir

def prepare_train_loaders(config):
    svhn_root = os.path.join(config['location'], 'svhn_data')
    image_folder_root = _convert_svhn_mat_to_imagefolder(svhn_root)
    train_dataset = datasets.ImageFolder(os.path.join(image_folder_root, 'train'), transform=config['train_preprocess'])
    return {'full': torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)}

def prepare_test_loaders(config):
    svhn_root = os.path.join(config['location'], 'svhn_data')
    image_folder_root = _convert_svhn_mat_to_imagefolder(svhn_root)
    test_dataset = datasets.ImageFolder(os.path.join(image_folder_root, 'test'), transform=config['eval_preprocess'])
    loaders = {'test': torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)}
    if config.get('val_fraction', 0) > 0:
        print('Splitting SVHN test set for validation')
        val_subset, test_subset = create_heldout_split(loaders['test'].dataset, fraction=config['val_fraction'])
        loaders['val'] = torch.utils.data.DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        loaders['test'] = torch.utils.data.DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    loaders['class_names'] = test_dataset.classes
    return loaders