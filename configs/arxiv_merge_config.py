import os
from .arxiv_classification_shared import (
    TASK_A_SUBSET, TASK_B_SUBSET,
    LABEL_TO_ID, ID_TO_LABEL, NUM_LABELS, ALL_LABELS
)

CWD = os.getcwd()
CACHE_DIR = "."
MODEL_DIR = "./lora_adapters_arxiv"
INGREDIENTS_PATH = os.path.join(CWD, "arxiv_merge_ingredients.pt")

config = {
    'dataset': [
        {'name': 'arxiv_cs', 'path': 'train_task_A_cs.csv', 'type': 'arxiv_multilabel', 'batch_size': 4, 'label_subset': TASK_A_SUBSET},
        
        {'name': 'arxiv_math', 'path': 'train_task_B_math.csv', 'type': 'arxiv_multilabel', 'batch_size': 4, 'label_subset': TASK_B_SUBSET},
        
        {'name': 'arxiv_combined', 'path': 'train_task_C_combined.csv', 'type': 'arxiv_multilabel', 'batch_size': 4, 'label_subset': ALL_LABELS},
        
        {'name': 'arxiv_union', 'path': 'train_task_D_union.csv', 'type': 'arxiv_multilabel', 'batch_size': 4, 'label_subset': ALL_LABELS},
    ],
    
    'evaluation_dataset': {
        'name': 'arxiv_curated_eval',
        'type': 'arxiv_multilabel',
        'path': 'arxiv_curated_eval.csv', 
        'batch_size': 8,
    },

    'model': {
        'name' : 'meta-llama/Llama-3.2-3B',
        'cachedir': CACHE_DIR,
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL, 
        'num_labels': NUM_LABELS,
        'ft_config': {'type': 'lora'},
        'peft_config': {
            'task_type' : "SEQ_CLS",
            'inference_mode' : False,
            'r': 16,
            'lora_alpha' : 16,
            'lora_dropout' : 0.1,
            'target_modules' : ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
        'bases': [
            f'{MODEL_DIR}/arxiv_cs_lora',
            f'{MODEL_DIR}/arxiv_math_lora',
        ],
    },
    
    'task_merge_config': {
        'ingredients_path' : INGREDIENTS_PATH,
        'representation': 'svd-vector',
        'merge_method': 'ties',
        'merging_type': 'mean',
        'scaling_coeffs': [0.5],
    },
    
    'eval_type': 'multi_label_accuracy',
}