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
from dataset.curated_reddit import prepare_curated_loaders
from configs.reddit_classification_shared import (
    ID_TO_SUBREDDIT
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

TOP_K=5

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

def evaluate_dual_label_accuracy(model, loader, device, top_k=TOP_K):
    model.to(device)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating Top-{top_k} Dual-Label Accuracy", leave=False):
            multi_hot_labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            
            top_k_preds_indices = torch.topk(outputs.logits, k=top_k, dim=-1).indices

            for i in range(len(inputs['input_ids'])):
                predicted_indices_set = set(top_k_preds_indices[i].cpu().numpy())
                
                true_indices_tensor = torch.where(multi_hot_labels[i] == 1)[0]
                true_indices_set = set(true_indices_tensor.cpu().numpy())
                
                if true_indices_set.issubset(predicted_indices_set):
                    correct_predictions += 1
            
            total_samples += len(inputs['input_ids'])

    if total_samples == 0: return 0.0
    return (correct_predictions / total_samples) * 100


def run_reddit_evaluation():
    CONFIG_NAME = 'reddit_merge_classification_config'
    COMPUTE_TRANSFORM = False
    HEADS_PATH = "reddit_heads_2_tasks.pt"
    
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

    print("Preparing experiment config (loading base models)...")
    config = prepare_experiment_config(raw_config)

    print("Preparing curated dual-label evaluation loader...")
    curated_eval_config = raw_config['curated_eval_dataset']
    curated_eval_config['tokenizer'] = tokenizer
    curated_eval_config['subreddit_to_id'] = subreddit_to_id
    curated_eval_config['num_workers'] = 0
    curated_loader = prepare_curated_loaders(curated_eval_config)['test']

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

        print(f'\n--- Evaluating Merged Backbone on Curated Dual-Label Task ---')
        
        head_types_to_test = {
            "SVD_MERGED": svd_merged_head_sd,
            "AVERAGED": avg_merged_head_sd,
        }

        for head_name, head_sd in head_types_to_test.items():
            print(f"\n  Injecting '{head_name}' head for evaluation...")
            merged_model.score.load_state_dict(head_sd, strict=True)
            
            dual_label_accuracy = evaluate_dual_label_accuracy(
                merged_model, curated_loader, device, top_k=TOP_K
            )
            print(f"  > Top-{TOP_K} Dual-Label Accuracy (with {head_name} Head): {dual_label_accuracy:.2f}%")
            all_results[f'top-{TOP_K}_dual_label_accuracy_{head_name}_head'] = dual_label_accuracy

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