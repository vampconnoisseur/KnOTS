"""
Configuration file for the Reddit subreddit classification and merging experiment.

This file defines all the parameters needed to run the training, merging, and
evaluation pipeline, including dataset specifications, model details, and merge
hyperparameters.
"""
import os
from .reddit_classification_shared import (
    LIFESTYLE_SUBREDDITS, SCIENCE_TECH_SUBREDDITS, ALL_SUBREDDITS,
    SUBREDDIT_TO_ID, ID_TO_SUBREDDIT, NUM_LABELS
)

CWD = os.getcwd()
DATA_DIR = os.path.join(CWD, "data")
MODEL_DIR = os.path.join(CWD, "lora_adapters_reddit")
HEADS_DIR = os.path.join(CWD, "head_layers")
INGREDIENTS_PATH = os.path.join(DATA_DIR, "reddit_merge_ingredients.pt")
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"

MIXED_SUBREDDITS = LIFESTYLE_SUBREDDITS + SCIENCE_TECH_SUBREDDITS

config = {
    'dataset': [
        {
            'name': 'reddit_lifestyle_culture', 'type': 'pushshift_reddit',
            'subreddits': LIFESTYLE_SUBREDDITS, 'batch_size': 4,
            'num_train_samples': 100000, 'num_val_samples': 10000,
        },
        {
            'name': 'reddit_science_culture', 'type': 'pushshift_reddit',
            'subreddits': SCIENCE_TECH_SUBREDDITS, 'batch_size': 4,
            'num_train_samples': 100000, 'num_val_samples': 10000,
        },
        {
            'name': 'reddit_mixed_culture', 'type': 'pushshift_reddit',
            'subreddits': MIXED_SUBREDDITS, 'batch_size': 4,
            'num_train_samples': 200000, 'num_val_samples': 10000, 
        },
    ],
    
    'curated_eval_dataset': {
        'name': 'reddit_curated_eval', 'type': 'curated_dual_label',
        'path': os.path.join(DATA_DIR, 'curated_dual_label_dataset.jsonl'), 'batch_size': 8,
    },

    'model': {
        'name' : BASE_MODEL_NAME, 'cachedir': ".", 'subreddit_to_id': SUBREDDIT_TO_ID,
        'id_to_label': ID_TO_SUBREDDIT, 'num_labels': NUM_LABELS,
        'bases': [
            f'{MODEL_DIR}/reddit_lifestyle_culture_lora',
            f'{MODEL_DIR}/reddit_science_culture_lora',
        ],
        'ft_config': {'type': 'lora'},
        'peft_config': {
            'task_type' : "SEQ_CLS", 'inference_mode' : False, 'r': 16,
            'lora_alpha' : 16, 'lora_dropout' : 0.1,
            'target_modules' : ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
    },

    'task_merge_config': {
        'ingredients_path' : INGREDIENTS_PATH, 'representation': 'svd-vector',
        'merge_method': 'ties', 'merging_type': 'mean', 'scaling_coeffs': [0.5],
    },
    
    'heads_path': os.path.join(HEADS_DIR, 'reddit_heads.pt'),
    'model_dir': MODEL_DIR,
    'eval_type': 'classification_accuracy',
}