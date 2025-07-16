import os
import torch
import numpy as np
from tqdm.auto import tqdm
import transformers
from huggingface_hub import login
from copy import deepcopy
from collections import OrderedDict

from utils import get_config_from_name, prepare_experiment_config, write_to_csv, set_seed
from task_merger import get_merge_handler
from ft_handlers import LoRAScoreHandler
from dataset.pushshift_reddit import prepare_test_loaders
from configs.reddit_classification_shared import LIFESTYLE_SUBREDDITS, SCIENCE_TECH_SUBREDDITS, ID_TO_SUBREDDIT

os.environ["TOKENIZERS_PARALLELISM"] = "true"
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

try:
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token: login(token=token)
except Exception: pass

class StateDictModel(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self._sd = sd

    def state_dict(self):
        return self._sd

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

    print("Preparing mixed-domain evaluation loader (ELI5)...")
    mixed_eval_config = raw_config['mixed_eval_dataset']
    mixed_eval_config['tokenizer'] = tokenizer
    mixed_eval_config['subreddit_to_id'] = subreddit_to_id
    mixed_eval_config['num_workers'] = 0
    mixed_domain_loader = prepare_test_loaders(mixed_eval_config)['test']

    print(f"Loading task-specific heads from {HEADS_PATH}...")
    task_heads = torch.load(HEADS_PATH, weights_only=True)

    csv_file = os.path.join('./csvs_reddit_classification_eval', f'final_merged_results.csv')
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f'Saving results to {csv_file}')

    def merge_and_eval(BackboneMerge, svd_merged_head_sd, avg_merged_head_sd, individual_heads_sd):
        instance_params = raw_config['task_merge_config']
        all_results = instance_params.copy()

        print('Merging models (Backbone Only)...')
        merged_model = BackboneMerge.merge(instance_params)

        print(f'\n--- Evaluating Merged Backbone on Mixed-Domain Task (ELI5) ---')

        print("\n  (1/4) Injecting SVD MERGED head for evaluation...")
        merged_model.score.load_state_dict(svd_merged_head_sd, strict=True)
        dual_accuracy_svd_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with SVD Merged Head): {dual_accuracy_svd_head:.2f}%")
        all_results['top5_dual_domain_accuracy_svd_head'] = dual_accuracy_svd_head

        print("\n  (2/4) Injecting AVERAGED head for evaluation...")
        merged_model.score.load_state_dict(avg_merged_head_sd, strict=True)
        dual_accuracy_avg_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with Averaged Head): {dual_accuracy_avg_head:.2f}%")
        all_results['top5_dual_domain_accuracy_avg_head'] = dual_accuracy_avg_head

        print("\n  (3/4) Injecting SCIENCE head for evaluation...")
        merged_model.score.load_state_dict(individual_heads_sd['reddit_science_culture'])
        dual_accuracy_science_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with Science Head): {dual_accuracy_science_head:.2f}%")
        all_results['top5_dual_domain_accuracy_science_head'] = dual_accuracy_science_head

        print("\n  (4/4) Injecting LIFESTYLE head for evaluation...")
        merged_model.score.load_state_dict(individual_heads_sd['reddit_lifestyle_culture'])
        dual_accuracy_lifestyle_head = evaluate_dual_domain_hits(merged_model, mixed_domain_loader, device, top_k=5)
        print(f"  > Top-5 Dual Domain Hit Rate (with Lifestyle Head): {dual_accuracy_lifestyle_head:.2f}%")
        all_results['top5_dual_domain_accuracy_lifestyle_head'] = dual_accuracy_lifestyle_head

        write_to_csv(all_results, csv_file)
        return all_results

    with torch.no_grad():
        backbone_merge_config = config['task_merge_config']
        MergeClass = get_merge_handler(backbone_merge_config['representation'])

        print("Instantiating Merge Handler for Backbones...")
        BackboneMerge = MergeClass(
            finetuned_models=np.array([m.cpu().eval() for m in config['models']['bases']]),
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=backbone_merge_config,
        )

        if COMPUTE_TRANSFORM and backbone_merge_config.get('ingredients_path'):
            print("Computing and saving SVD ingredients for backbones...")
            BackboneMerge.transform(backbone_merge_config)
            print(f"Ingredients saved to {backbone_merge_config['ingredients_path']}")

        print("\nInstantiating Merge Handler for SVD Head Merge...")
        head_merge_config = deepcopy(backbone_merge_config)
        head_merge_config['ingredients_path'] = None 

        head_models = [StateDictModel(sd) for sd in task_heads.values()]
        
        template_head_sd = next(iter(task_heads.values()))
        zero_head_sd = OrderedDict()
        for key, tensor in template_head_sd.items():
            zero_head_sd[key] = torch.zeros_like(tensor)
        zero_reference_head_model = StateDictModel(zero_head_sd)

        HeadMerge = MergeClass(
            finetuned_models=np.array(head_models),
            pretrained_model=zero_reference_head_model, 
            param_handler=LoRAScoreHandler,
            device=device,
            merge_config=head_merge_config,
        )
        
        print("Computing SVD and merging classification heads...")
        HeadMerge.transform(head_merge_config)
        final_svd_head_sd = HeadMerge.merge(head_merge_config, return_state_dict_only=True)
        print("SVD Head merge complete.")
        
        print("\nCalculating simple averaged head...")
        template_tensor = next(iter(task_heads.values()))['original_module.weight']
        summed_original_weights = torch.zeros_like(template_tensor)
        summed_trainable_weights = torch.zeros_like(template_tensor)

        for head_sd in task_heads.values():
            summed_original_weights += head_sd['original_module.weight']
            summed_trainable_weights += head_sd['modules_to_save.default.weight']
        
        num_heads = len(task_heads)
        averaged_original_weight = summed_original_weights / num_heads
        averaged_trainable_weight = summed_trainable_weights / num_heads
        
        final_avg_head_sd = OrderedDict()
        final_avg_head_sd['original_module.weight'] = averaged_original_weight
        final_avg_head_sd['modules_to_save.default.weight'] = averaged_trainable_weight
        print("Averaged head calculation complete.")
        
        final_results = merge_and_eval(BackboneMerge, final_svd_head_sd, final_avg_head_sd, task_heads)
        print("\n--- Final Results ---")
        print(final_results)

if __name__ == "__main__":
    run_reddit_evaluation()