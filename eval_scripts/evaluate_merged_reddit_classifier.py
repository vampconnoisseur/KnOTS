import os
import torch
import numpy as np
from tqdm.auto import tqdm
import transformers
from huggingface_hub import login

from utils import get_config_from_name, prepare_experiment_config, write_to_csv, set_seed
from task_merger import get_merge_handler
from dataset.pushshift_reddit import prepare_test_loaders
from configs.reddit_classification_shared import LIFESTYLE_SUBREDDITS, SCIENCE_TECH_SUBREDDITS, ID_TO_SUBREDDIT

os.environ["TOKENIZERS_PARALLELISM"] = "true"
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

try:
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token: login(token=token)
except Exception: pass

def evaluate_standard_accuracy(model, loader, device):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Standard Accuracy", leave=False):
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if not all_labels: return 0.0
    return np.mean(np.array(all_preds) == np.array(all_labels))


def evaluate_dual_domain_hits(model, loader, device, top_k=5):
    model.to(device)
    model.eval()

    science_set = set(SCIENCE_TECH_SUBREDDITS)
    lifestyle_set = set(LIFESTYLE_SUBREDDITS)

    dual_domain_hits = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating Top-{top_k} Dual Domain Hits", leave=False):
            batch.pop('labels', None)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            
            top_k_preds_indices = torch.topk(outputs.logits, k=top_k, dim=-1).indices

            for single_example_preds in top_k_preds_indices:
                predicted_subreddits = {ID_TO_SUBREDDIT[idx.item()] for idx in single_example_preds}
                
                has_science_hit = not predicted_subreddits.isdisjoint(science_set)
                has_lifestyle_hit = not predicted_subreddits.isdisjoint(lifestyle_set)

                if has_science_hit and has_lifestyle_hit:
                    dual_domain_hits += 1
            
            total_samples += len(inputs['input_ids'])

    if total_samples == 0: return 0.0
    return (dual_domain_hits / total_samples) * 100


def run_reddit_evaluation():
    CONFIG_NAME = 'reddit_merge_classification_config'
    COMPUTE_TRANSFORM = False
    HEADS_PATH = "reddit_heads.pt"
    
    SEED = 123
    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = get_config_from_name(CONFIG_NAME, device=device)

    # --- 1. PREPARE ALL CONFIGS AND DATA FIRST ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(raw_config['model']['name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.sep_token is None: tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    subreddit_to_id = raw_config['model']['subreddit_to_id']
    
    for d_config in raw_config['dataset']:
        d_config['tokenizer'] = tokenizer
        d_config['subreddit_to_id'] = subreddit_to_id
        d_config['num_workers'] = 0

    print("Preparing experiment config (models and expert test data)...")
    config = prepare_experiment_config(raw_config)

    original_random_head_sd = config['models']['new'].score.state_dict()

    print("Preparing mixed-domain evaluation loader (ELI5)...")
    mixed_eval_config = raw_config['mixed_eval_dataset']
    mixed_eval_config['tokenizer'] = tokenizer
    mixed_eval_config['subreddit_to_id'] = subreddit_to_id
    mixed_eval_config['num_workers'] = 0 # Ensure no multiprocessing issues
    mixed_domain_loader = prepare_test_loaders(mixed_eval_config)['test']

    print(f"Loading task-specific heads from {HEADS_PATH}...")
    task_heads = torch.load(HEADS_PATH)

    csv_file = os.path.join('./csvs_reddit_classification_eval', f'final_merged_results.csv')
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f'Saving results to {csv_file}')

    def merge_and_eval(Merge, random_head_sd):
        instance_params = raw_config['task_merge_config']
        all_results = instance_params.copy()

        print('Merging models (Backbone Only)...')
        merged_model = Merge.merge(instance_params)

        # --- Evaluation on Original "Expert" Tasks ---
        print(f'\n--- Evaluating Merged Backbone with Task-Specific Heads ---')
        dataset_names = [i['name'] for i in raw_config['dataset']]
        for i, data_cfg in enumerate(config['data']):
            loader = data_cfg['test']['test']
            task_name = dataset_names[i]
            print(f"  Injecting head for task: {task_name}")
            merged_model.score.load_state_dict(task_heads[task_name])
            accuracy = evaluate_standard_accuracy(merged_model, loader, device)
            print(f"  > Accuracy on '{task_name}': {accuracy:.4f}")
            all_results[f'{task_name}_accuracy'] = accuracy
        
        # --- Evaluation on Mixed-Domain Task (ELI5) with DIFFERENT HEADS ---
        print(f'\n--- Evaluating Merged Model on Mixed-Domain Task (ELI5) ---')

        # === Test with SCIENCE Head ===
        print("\n  (Run 1/3) Injecting SCIENCE head for mixed-domain evaluation...")
        merged_model.score.load_state_dict(task_heads['reddit_science_culture'])
        dual_accuracy_science_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with Science Head): {dual_accuracy_science_head:.2f}%")
        all_results['top5_dual_domain_accuracy_science_head'] = dual_accuracy_science_head

        # === Test with LIFESTYLE Head ===
        print("\n  (Run 2/3) Injecting LIFESTYLE head for mixed-domain evaluation...")
        merged_model.score.load_state_dict(task_heads['reddit_lifestyle_culture'])
        dual_accuracy_lifestyle_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with Lifestyle Head): {dual_accuracy_lifestyle_head:.2f}%")
        all_results['top5_dual_domain_accuracy_lifestyle_head'] = dual_accuracy_lifestyle_head

        # === Test with RANDOM Head ===
        print("\n  (Run 3/3) Re-initializing head using original random state...")
        merged_model.score.load_state_dict(random_head_sd)
        dual_accuracy_random_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with Random Head): {dual_accuracy_random_head:.2f}%")
        all_results['top5_dual_domain_accuracy_random_head'] = dual_accuracy_random_head

        write_to_csv(all_results, csv_file)
        return all_results

    with torch.no_grad():
        print("Instantiating Merge Handler...")
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            finetuned_models=np.array([m.cpu().eval() for m in config['models']['bases']]),
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        if COMPUTE_TRANSFORM:
            print("Computing and saving SVD ingredients...")
            Merge.transform(raw_config['task_merge_config'])
            print(f"Ingredients saved to {raw_config['task_merge_config']['ingredients_path']}")

        final_results = merge_and_eval(Merge, original_random_head_sd)
        print("\n--- Final Results ---")
        print(final_results)

if __name__ == "__main__":
    run_reddit_evaluation()