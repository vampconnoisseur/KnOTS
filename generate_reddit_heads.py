import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

ADAPTER_PATHS = {
    # "reddit_lifestyle_culture": "./lora_rank16_4_tasks/reddit_lifestyle_culture_lora",
    "reddit_science_culture": "./lora_rank16_4_tasks/reddit_science_culture_lora",
    "reddit_gaming_culture": "./lora_rank16_4_tasks/reddit_gaming_culture_lora",
    "reddit_finance_culture": "./lora_rank16_4_tasks/reddit_finance_culture_lora",
    "reddit_automotive_culture": "./lora_rank16_4_tasks/reddit_automotive_culture_lora",
    # "reddit_hobbies_culture": "./lora_rank16/reddit_hobbies_culture_lora",
}
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B"
from configs.reddit_classification_shared import NUM_LABELS 
SAVE_PATH = "reddit_heads_4_tasks.pt"

if __name__ == "__main__":
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    
    extracted_heads = {}
    for task_name, adapter_path in ADAPTER_PATHS.items():
        print(f"Extracting head for task: {task_name} from {adapter_path}")
        
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        head_state_dict = peft_model.score.state_dict()
        
        extracted_heads[task_name] = {k: v.cpu().clone() for k, v in head_state_dict.items()}

    torch.save(extracted_heads, SAVE_PATH)
    print(f"\nSuccessfully extracted and saved {len(extracted_heads)} heads to '{SAVE_PATH}'")

    print("\nVerifying saved heads:")
    loaded_heads = torch.load(SAVE_PATH)
    for task_name, head_sd in loaded_heads.items():
        print(f"  Task: {task_name}")
        for param_name, tensor in head_sd.items():
            print(f"    - {param_name}: {tensor.shape}")