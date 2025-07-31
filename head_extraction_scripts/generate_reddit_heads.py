"""
Extracts and saves the classification head from trained LoRA adapters.

This script loads a base sequence classification model and then, for each specified
LoRA adapter path, it applies the adapter and extracts the state dictionary of the
final classification layer (named 'score'). All extracted heads are saved into a
single .pt file for later use in model merging experiments.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
import os
from utils import get_config_from_name

if __name__ == "__main__":
    CONFIG_NAME = 'reddit_merge_classification_config'
    config = get_config_from_name(CONFIG_NAME)
    
    MODEL_DIR = config['model_dir']
    HEADS_PATH = config['heads_path']
    BASE_MODEL_NAME = config['model']['name']
    NUM_LABELS = config['model']['num_labels']
    
    os.makedirs(os.path.dirname(HEADS_PATH), exist_ok=True)

    ADAPTER_PATHS = {
        "reddit_lifestyle": os.path.join(MODEL_DIR, "reddit_lifestyle_culture_lora"),
        "reddit_science": os.path.join(MODEL_DIR, "reddit_science_culture_lora"),
    }

    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, num_labels=NUM_LABELS
    )
    
    extracted_heads = {}
    for task_name, adapter_path in ADAPTER_PATHS.items():
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter path not found, skipping: {adapter_path}")
            continue
        print(f"Extracting head for task: {task_name} from {adapter_path}")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        head_state_dict = peft_model.score.state_dict()
        extracted_heads[task_name] = {k: v.cpu().clone() for k, v in head_state_dict.items()}

    torch.save(extracted_heads, HEADS_PATH)
    print(f"\nSuccessfully extracted and saved {len(extracted_heads)} heads to '{HEADS_PATH}'")