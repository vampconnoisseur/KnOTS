import json
import torch
from torch.utils.data import DataLoader, Dataset

class CuratedDualLabelDataset(Dataset):
    def __init__(self, jsonl_file_path, tokenizer, subreddit_to_id, max_length=256):
        self.tokenizer = tokenizer
        self.subreddit_to_id = subreddit_to_id
        self.max_length = max_length
        self.data = []
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        title = item.get('title', '')
        selftext = item.get('selftext', '')

        if not isinstance(title, str): title = ""
        if not isinstance(selftext, str): selftext = ""

        text = f"Title: {title} {self.tokenizer.sep_token} {selftext}"

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        subreddit_labels = item.get('labels', [])
        
        multi_hot_labels = torch.zeros(len(self.subreddit_to_id), dtype=torch.float)
        for label_name in subreddit_labels:
            if label_name in self.subreddit_to_id:
                label_id = self.subreddit_to_id[label_name]
                multi_hot_labels[label_id] = 1.0

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': multi_hot_labels
        }

def prepare_curated_loaders(config):
    dataset = CuratedDualLabelDataset(
        jsonl_file_path=config['path'],
        tokenizer=config['tokenizer'],
        subreddit_to_id=config['subreddit_to_id']
    )
    
    test_loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        num_workers=config.get('num_workers', 0),
        shuffle=False
    )
    
    return {'test': test_loader}