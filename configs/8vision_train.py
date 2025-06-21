import os

# VIT_ARCH = 'ViT-L-14-CLIP'    # Model Architecture (Uncomment for ViT-L-14-CLIP)
VIT_ARCH = 'ViT-B-32-CLIP'      # Model Architecture (Uncomment for ViT-B-32-CLIP)
MODEL_DIR = 'lora_rank16'
# Cache for models and datasets will be in the project root '.'
CACHE_DIR = '' 
# Heads for text embeddings will be in './ViT-B-32-CLIP/'
HEAD_DIR = ''

if VIT_ARCH == 'ViT-L-14-CLIP':
    BASE_TYPE = "openai/clip-vit-large-patch14"
elif VIT_ARCH == 'ViT-B-32-CLIP':
    BASE_TYPE = "openai/clip-vit-base-patch32"

config = {
    'dataset': [
        {
            'name': 'stanford_cars',
            'type': 'cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
        },
        #  {
        #     'name': 'dtd',
        #     'type': 'dtd',
        #     'batch_size': 32,
        #     'num_workers': 16,
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'val_fraction': 0.2,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
        # },
        # {
        #     'name': 'eurosat',
        #     'type': 'eurosat', 
        #     'batch_size': 32,
        #     'num_workers': 16,
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'val_fraction': 0.2,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
        # },
        # {
        #     'name': 'gtsrb',
        #     'type': 'gtsrb', 
        #     'batch_size': 32,
        #     'num_workers': 16,
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'val_fraction': 0.2,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
        # },
        # {
        #     'name': 'sun397',
        #     'type': 'sun397',
        #     'batch_size': 32,
        #     'num_workers': 16,
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'val_fraction': 0.2,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'sun397_head.pt'),
        #     'hf_cache_dir': CACHE_DIR,
        # },
        # {
        #     'name': 'mnist',
        #     'type': 'mnist',
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
        #     'val_fraction': 0.2,
        #     'batch_size': 32,
        #     'num_workers': 16,
        # },
        # {
        #     'name': 'resisc45',
        #     'type': 'resisc45',
        #     'batch_size': 32,
        #     'num_workers': 16, 
        #     'shuffle_train': True,
        #     'crop_ratio': 1.0,
        #     'val_fraction': 0.2,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
        # },
        # {
        #     'name': 'svhn',
        #     'shuffle_train': True,
        #     'type': 'svhn',
        #     'crop_ratio': 1.0,
        #     'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
        #     'val_fraction': 0.2,
        #     'batch_size': 32,
        #     'num_workers': 8,
        #     'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/svhn_shuffled_idxs.pt')
        # },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': BASE_TYPE,
        'cachedir': CACHE_DIR,
        'bases': [],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'eval_type': 'clip',
}
