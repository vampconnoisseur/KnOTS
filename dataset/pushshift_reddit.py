import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

class HFPushshiftSequenceClassificationDataset(IterableDataset):
    """
    An iterable PyTorch Dataset for streaming Reddit data, formatted for
    Sequence Classification.
    """
    def __init__(self, hf_split, tokenizer, subreddit_to_id, max_length=256):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subreddit_to_id = subreddit_to_id

    def __iter__(self):
        for item in self.hf_split:
            title = item.get('title', '')
            selftext = item.get('selftext', '')

            if not isinstance(title, str): title = ""
            if not isinstance(selftext, str): selftext = ""

            # Format for classification. Using the tokenizer's separator token is robust.
            text = f"Title: {title} {self.tokenizer.sep_token} {selftext}"

            encodings = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            subreddit = item.get('subreddit')
            # Yield data only if the subreddit is in our predefined map
            if subreddit and subreddit in self.subreddit_to_id:
                label = self.subreddit_to_id[subreddit]
                yield {
                    'input_ids': encodings['input_ids'].squeeze(0),
                    'attention_mask': encodings['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

def _prepare_loaders(config, split_name, num_samples, seed):
    """Helper function to create a data loader for a given split."""
    subreddits_to_use = config['subreddits']

    full_dataset_stream = load_dataset(
        "fddemarco/pushshift-reddit",
        split="train",
        streaming=True,
        cache_dir=config.get('hf_cache_dir')
    )

    filtered_stream = full_dataset_stream.filter(lambda ex: ex.get('subreddit') in subreddits_to_use)
    shuffled_stream = filtered_stream.shuffle(seed=seed, buffer_size=10000)
    data_stream = shuffled_stream.take(num_samples)

    dataset = HFPushshiftSequenceClassificationDataset(
        data_stream,
        config['tokenizer'],
        config['subreddit_to_id']
    )
    return DataLoader(dataset, batch_size=config['batch_size'], num_workers=config.get('num_workers', 0))

def prepare_train_loaders(config):
    train_loader = _prepare_loaders(
        config=config,
        split_name="train",
        num_samples=config.get('num_train_samples', 20000),
        seed=42
    )
    return {'full': train_loader}

def prepare_test_loaders(config):
    """Prepares validation and test DataLoaders."""
    num_val_samples = config.get('num_val_samples', 1000)
    num_test_samples = config.get('num_test_samples', 1000)

    val_loader = _prepare_loaders(
        config=config,
        split_name="train",
        num_samples=num_val_samples,
        seed=123 
    )
    test_loader = _prepare_loaders(
        config=config,
        split_name="train",
        num_samples=num_test_samples,
        seed=456
    )

    return {'val': val_loader, 'test': test_loader}