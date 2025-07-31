import os

VIT_ARCH = 'ViT-L-14-CLIP'  # Model Architecture
# IMPORTANT: This should be the directory containing your adapter files
# e.g., 'lora_adaptors_vision/stanford_cars_...pt'
MODEL_DIR = 'lora_adaptors_vision' 
CACHE_DIR = '.'              # Where to cache HF pretrained checkpoints
HEAD_DIR = '.'               # CLIP Head Directory

config = {
    'dataset': [
        {
            'name': 'stanford_cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'type': 'stanford_cars',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/cars_shuffled_idxs.pt')
        },
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'type': 'dtd',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
            'val_fraction': 0.5, 
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/dtd_shuffled_idxs.pt') 
        },
        {
            'name': 'eurosat',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'type': 'eurosat',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
            'val_fraction': 0.5, 
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/eurosat_shuffled_idxs.pt') 
        },
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'type': 'gtsrb',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/gtsrb_shuffled_idxs.pt')
        },
        {
            'name': 'sun397',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'type': 'sun397',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'sun397_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/sun397_shuffled_idxs.pt')
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'type': 'mnist',
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,  
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/mnist_shuffled_idxs.pt')
        },
        {
            'name': 'svhn',
            'shuffle_train': True,
            'type': 'svhn',
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/svhn_shuffled_idxs.pt')
        },
        {
            'name': 'resisc45',
            'shuffle_train': True,
            'type': 'resisc45',
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
            'batch_size': 32,
            'num_workers': 8,  
            'val_fraction': 0.5, 
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/resisc45_shuffled_idxs.pt')
        },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': "openai/clip-vit-large-patch14",
        'cachedir': CACHE_DIR,
        'bases': [
            f'{MODEL_DIR}/stanford_cars',  
            f'{MODEL_DIR}/dtd',
            f'{MODEL_DIR}/gtsrb',
            f'{MODEL_DIR}/sun397',
            f'{MODEL_DIR}/eurosat',  
            f'{MODEL_DIR}/mnist',
            f'{MODEL_DIR}/gtsrb',
            f'{MODEL_DIR}/svhn',
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'task_merge_config': {
        'representation': 'svd-vector',
        'sign_resolve_mode': 'sum_of_values',
        'scaling_coeffs': .5,
        'topK': 20,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'concat_across_output': True,
        'dare' : False,
        'dare_pruning_coeffs' : 0.0,
    },
    'eval_type': 'clip',
}
