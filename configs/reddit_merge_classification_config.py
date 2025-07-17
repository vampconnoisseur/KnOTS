# In configs/reddit_merge_classification_config.py
import os
from .reddit_classification_shared import (
    LIFESTYLE_SUBREDDITS, SCIENCE_TECH_SUBREDDITS, GAMING_SUBREDDITS,
    FINANCE_SUBREDDITS, AUTOMOTIVE_SUBREDDITS, HOBBIES_SUBREDDITS,
    SUBREDDIT_TO_ID, NUM_LABELS
)

CWD = os.getcwd()
CACHE_DIR = ""
MODEL_DIR = "./lora_rank16_2_tasks"
INGREDIENTS_PATH = os.path.join(CWD, "reddit_classification_ingredients_2_tasks.pt") 

config = {
    'dataset': [
        {'name': 'reddit_lifestyle_culture', 'subreddits': LIFESTYLE_SUBREDDITS, 'type': 'pushshift_reddit', 'batch_size': 8},
        {'name': 'reddit_science_culture', 'subreddits': SCIENCE_TECH_SUBREDDITS, 'type': 'pushshift_reddit', 'batch_size': 8},
        # {'name': 'reddit_gaming_culture', 'subreddits': GAMING_SUBREDDITS, 'type': 'pushshift_reddit', 'batch_size': 8},
        # {'name': 'reddit_finance_culture', 'subreddits': FINANCE_SUBREDDITS, 'type': 'pushshift_reddit', 'batch_size': 8},
        # {'name': 'reddit_automotive_culture', 'subreddits': AUTOMOTIVE_SUBREDDITS, 'type': 'pushshift_reddit', 'batch_size': 8},
    ],
    
    'curated_eval_dataset': {
        'name': 'reddit_curated_dual_label',
        'type': 'curated_dual_label',
        'path': './curated_dual_label_dataset.jsonl',
        'batch_size': 8,
    },

    'model': {
        'name' : 'meta-llama/Llama-3.2-3B',
        'cachedir': CACHE_DIR,
        'subreddit_to_id': SUBREDDIT_TO_ID,
        'num_labels': NUM_LABELS,
        'bases': [
            f'{MODEL_DIR}/reddit_science_culture_lora',
            f'{MODEL_DIR}/reddit_lifestyle_culture_lora',
            # f'{MODEL_DIR}/reddit_gaming_culture_lora',
            # f'{MODEL_DIR}/reddit_finance_culture_lora',
            # f'{MODEL_DIR}/reddit_automotive_culture_lora',
        ],
        'ft_config': {'type': 'lora'},
        'peft_config': {
            'task_type' : "SEQ_CLS",
            'inference_mode' : True,
            'r': 16,
            'lora_alpha' : 16,
            'lora_dropout' : 0.1,
            'target_modules' : ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
    },
    'task_merge_config': {
        'ingredients_path' : INGREDIENTS_PATH,
        'representation': 'svd-vector',
        'sign_resolve_mode': 'sum_of_values',
        'topK': 100,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'scaling_coeffs': [1/2],
        'concat_across_output': True,
        'dare' : False,
        'dare_pruning_coeffs': 0.0
    },
    'eval_type': 'classification_accuracy',
}