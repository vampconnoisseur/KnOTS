# In configs/reddit_mixed_classification_config.py

from .reddit_classification_shared import (
    SUBREDDIT_TO_ID,
    NUM_LABELS,
    LIFESTYLE_SUBREDDITS,
    SCIENCE_TECH_SUBREDDITS
)
import os
CACHE_DIR = ''

MIXED_SUBREDDITS = LIFESTYLE_SUBREDDITS + SCIENCE_TECH_SUBREDDITS

config = {
    'dataset': [{
        'name': 'reddit_mixed_culture',
        'type': 'pushshift_reddit',
        'batch_size': 4,
        'hf_cache_dir': CACHE_DIR,
        'subreddits': MIXED_SUBREDDITS,
        'num_train_samples': 100000,
        'num_val_samples': 10000,
        'num_test_samples': 10000,
    }],
    'model': {
        'base_type': "meta-llama/Llama-3.2-3B",
        'cachedir': CACHE_DIR,
        'subreddit_to_id': SUBREDDIT_TO_ID,
        'num_labels': NUM_LABELS,
        'ft_config': {
            'r': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'bias': "none",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    },
    'eval_type': 'sequence_classification'
}