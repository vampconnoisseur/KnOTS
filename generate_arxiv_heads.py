import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
import os

from configs.arxiv_classification_shared import NUM_LABELS

ADAPTER_PATHS = {
    "arxiv_cs": "./lora_adapters_arxiv/arxiv_cs_lora",
    "arxiv_math": "./lora_adapters_arxiv/arxiv_math_lora",
}
BASE_MODEL_NAME = 'meta-llama/Llama-3.2-3B' 
SAVE_PATH = "arxiv_heads.pt"

if __name__ == "__main__":
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, 
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    
    extracted_heads = {}
    for task_name, adapter_path in ADAPTER_PATHS.items():
        if not os.path.exists(adapter_path):
            print(f"Warning: Adapter path not found, skipping: {adapter_path}")
            continue
            
        print(f"Extracting head for task: {task_name} from {adapter_path}")
        
        model_for_loading = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
        )
        peft_model = PeftModel.from_pretrained(model_for_loading, adapter_path)
        
        head_state_dict = peft_model.score.state_dict()
        
        extracted_heads[task_name] = {k: v.cpu().clone() for k, v in head_state_dict.items()}
        print(f"  - Extracted head with keys: {list(head_state_dict.keys())}")

    if not extracted_heads:
        print("\nNo heads were extracted. Please check the ADAPTER_PATHS.")
    else:
        torch.save(extracted_heads, SAVE_PATH)
        print(f"\nSuccessfully extracted and saved {len(extracted_heads)} heads to '{SAVE_PATH}'")

        print("\nVerifying saved heads:")
        loaded_heads = torch.load(SAVE_PATH)
        for task_name, head_sd in loaded_heads.items():
            print(f"  Task: {task_name}")
            for param_name, tensor in head_sd.items():
                print(f"    - {param_name}: {tensor.shape}")