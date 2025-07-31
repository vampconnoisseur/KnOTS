"""
Defines an iterable Dataset and DataLoaders for streaming Reddit data.

This module provides tools to stream data from the fddemarco/pushshift-reddit
dataset on the Hugging Face Hub, filter it by a list of subreddits, and format
it for sequence classification tasks.
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

class HFPushshiftSequenceClassificationDataset(IterableDataset):
    """
    An iterable PyTorch Dataset for streaming Reddit data for sequence classification.

    This dataset streams items from a Hugging Face dataset split, filters them
    by subreddit, and formats the title and selftext for a classification model.

    Args:
        hf_split (datasets.iterable_dataset.IterableDataset): The dataset split
            from `load_dataset` with `streaming=True`.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding text.
        subreddit_to_id (dict): Mapping from subreddit names to integer class IDs.
        max_length (int, optional): The maximum sequence length for tokenization.
    """
    def __init__(self, hf_split, tokenizer, subreddit_to_id, max_length=256):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subreddit_to_id = subreddit_to_id

    def __iter__(self):
        """
        Yields tokenized text and a corresponding label for each item in the stream.
        """
        for item in self.hf_split:
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
    """
    Helper function to create a data loader for a given split from the Reddit stream.

    Args:
        config (dict): Configuration dictionary with parameters like 'subreddits'.
        split_name (str): The name of the dataset split (e.g., 'train').
        num_samples (int): The number of samples to take from the stream.
        seed (int): The random seed for shuffling the stream.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the prepared dataset.
    """
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
    """
    Prepares a DataLoader for the training set.

    Args:
        config (dict): A configuration dictionary with training set parameters.

    Returns:
        dict: A dictionary containing the training DataLoader, e.g., {'full': loader}.
    """
    train_loader = _prepare_loaders(
        config=config,
        split_name="train",
        num_samples=config.get('num_train_samples', 100000),
        seed=42
    )

    return {'full': train_loader}

def prepare_test_loaders(config):
    """
    Prepares validation and test DataLoaders.

    Args:
        config (dict): A configuration dictionary with validation/test set parameters.

    Returns:
        dict: A dictionary containing 'val' and 'test' DataLoaders.
    """
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