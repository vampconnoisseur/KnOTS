from .arxiv_classification_shared import NUM_LABELS, LABEL_TO_ID, ALL_LABELS

config = {
    'model': {
        'base_type': 'meta-llama/Meta-Llama-3-8B',
        'cachedir': '.',
        'num_labels': NUM_LABELS,
        'label_map': LABEL_TO_ID,
        'problem_type': "multi_label_classification",
        'ft_config': {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
    },
    'dataset': {
        'data_path': './arxiv-multi-label-text-classification-datasets/arxiv_data.csv',
        'name': 'arxiv_cs_multi_label',
        'labels_to_use': set(ALL_LABELS),
        'label_to_id': LABEL_TO_ID,
        'batch_size': 4,
        'max_length': 512,
        'num_train_samples': 20000,
        'num_val_samples': 2000,
    }
}
