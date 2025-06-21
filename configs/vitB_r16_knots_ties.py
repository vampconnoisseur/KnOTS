import os

VIT_ARCH = 'ViT-B-32-CLIP'
MODEL_DIR = 'lora_rank16'
CACHE_DIR = ''
HEAD_DIR = '' 
DATA_DIR = 'data' 


config = {
    'dataset': [
        {
            'name': 'stanford_cars', 'type': 'cars', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'location': DATA_DIR, 
        },
        {
            'name': 'dtd', 'type': 'dtd', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'hf_cache_dir': CACHE_DIR, 
        },
        {
            'name': 'eurosat', 'type': 'eurosat', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
            'hf_cache_dir': CACHE_DIR,
        },
        {
            'name': 'gtsrb', 'type': 'gtsrb', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
            'hf_cache_dir': CACHE_DIR,
        },
        {
            'name': 'mnist', 'type': 'mnist', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'location': DATA_DIR, 
        },
        {
            'name': 'resisc45', 'type': 'resisc45', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
            'hf_cache_dir': CACHE_DIR,
        },
        {
            'name': 'sun397', 'type': 'sun397', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'sun397_head.pt'),
            'hf_cache_dir': CACHE_DIR,
        },
        {
            'name': 'svhn', 'type': 'svhn', 'batch_size': 32, 'num_workers': 16,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
            'location': DATA_DIR,
        },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': "openai/clip-vit-base-patch32",
        'cachedir': CACHE_DIR,
        
        'bases': [
            os.path.join(MODEL_DIR, 'stanford_cars.pt'),
            os.path.join(MODEL_DIR, 'dtd.pt'),
            os.path.join(MODEL_DIR, 'eurosat.pt'),
            os.path.join(MODEL_DIR, 'gtsrb.pt'),
            os.path.join(MODEL_DIR, 'mnist.pt'),
            os.path.join(MODEL_DIR, 'resisc45.pt'),
            os.path.join(MODEL_DIR, 'sun397.pt'),
            os.path.join(MODEL_DIR, 'svhn.pt'),
        ],
        'ft_config': {
            'type': 'lora', 'r': 16, 'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1, 'bias': "none",
        },
    },
    'task_merge_config': {
        'representation': 'svd-vector',
        'sign_resolve_mode': 'sum_of_values',
        'scaling_coeffs': 0.6,
        'topK': 20,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'concat_across_output': True,
        'dare' : False,
        'dare_pruning_coeffs' : 0.0,
    },
    'eval_type': 'clip'
}