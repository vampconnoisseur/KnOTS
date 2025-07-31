"""
Defines the Dataset and DataLoader logic for the ArXiv multi-label classification task.

This script provides the necessary components to read preprocessed ArXiv CSV files,
tokenize the text content (abstracts), and convert the category labels into a
multi-hot encoded tensor format suitable for training a multi-label classifier.

Key components:
- clean_and_extract_labels: A utility function to parse the raw category strings.
- ArxivMultiLabelDataset: A PyTorch Dataset class that handles the core logic of
  loading, tokenizing, and vectorizing individual data samples.
- prepare_dataloaders: A high-level function that takes a configuration dictionary,
  splits data into training and validation sets, and returns the corresponding
  PyTorch DataLoaders.
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import re
from sklearn.model_selection import train_test_split
import os

def clean_and_extract_labels(label_string):
    """
    Cleans a raw string of ArXiv category labels into a clean list.

    Handles various separators (commas, semicolons, spaces) and removes
    unwanted characters.

    Args:
        label_string (str): The raw string from the 'categories' column.

    Returns:
        list[str]: A list of unique, cleaned labels found in the string.
    """
    if not isinstance(label_string, str):
        return []
    s = re.sub(r'[;,]', ' ', label_string)
    s = re.sub(r"[\[\]'()\"]", '', s)
    potential_labels = s.split()
    # A simple heuristic to filter for valid ArXiv category codes
    cleaned_labels = [label for label in potential_labels if '.' in label or '-' in label or label.islower()]
    return cleaned_labels

class ArxivMultiLabelDataset(Dataset):
    """
    A PyTorch Dataset for the ArXiv multi-label classification task.

    Reads a CSV file containing ArXiv paper abstracts and their categories,
    and prepares them for multi-label classification.

    Attributes:
        df (pd.DataFrame): The pandas DataFrame holding the dataset.
        tokenizer: The Hugging Face tokenizer instance.
        label_to_id (dict): A mapping from category string to integer ID.
        max_length (int): The maximum sequence length for tokenization.
        num_labels (int): The total number of unique labels in the classification task.
        data (list[dict]): The dataset converted to a list of records for fast access.
    """
    def __init__(self, csv_path, tokenizer, label_to_id, max_length=512, label_subset=None):
        """
        Initializes the ArxivMultiLabelDataset.

        Args:
            csv_path (str): The path to the input CSV file.
            tokenizer: The Hugging Face tokenizer to use for text processing.
            label_to_id (dict): A dictionary mapping label names to integer indices.
            max_length (int, optional): Maximum token length for model input. Defaults to 512.
            label_subset (list[str], optional): A list of labels to filter the dataset by.
                                                If provided, only samples containing at least
                                                one of these labels will be included. Defaults to None.
        """
        self.df = pd.read_csv(csv_path).dropna(subset=['Abstracts', 'categories'])
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.num_labels = len(label_to_id)

        if label_subset:
            subset_set = set(label_subset)
            mask = self.df['categories'].apply(lambda cats: any(c in subset_set for c in clean_and_extract_labels(cats)))
            self.df = self.df[mask].reset_index(drop=True)

        self.data = self.df.to_dict('records')

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves one sample from the dataset at the specified index.

        The method tokenizes the abstract text and converts the category labels into a
        multi-hot encoded tensor.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        item = self.data[idx]
        text = str(item.get('Abstracts', ''))

        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item_labels = clean_and_extract_labels(item.get('categories', ''))
        multi_hot_labels = torch.zeros(self.num_labels, dtype=torch.float)
        for label_name in item_labels:
            if label_name in self.label_to_id:
                label_id = self.label_to_id[label_name]
                multi_hot_labels[label_id] = 1.0

        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': multi_hot_labels
        }


def prepare_dataloaders(config):
    """
    Creates and returns train and validation/test DataLoaders for an ArXiv dataset.

    This function handles two cases:
    1. If the config is for a final evaluation set ('arxiv_curated_eval'), it
       creates a single test DataLoader.
    2. For training configurations, it loads the specified training CSV, splits it
       into a 90% training set and a 10% validation set, and returns DataLoaders
       for both.

    Args:
        config (dict): A configuration dictionary containing all necessary parameters,
                       including 'path', 'tokenizer', 'label_to_id', 'batch_size', etc.

    Returns:
        dict: A dictionary of DataLoaders. For training, it contains {'train': ..., 'test': ...}.
              For evaluation, it contains {'union_eval': ...}.
    """
    if config.get('name') == 'arxiv_curated_eval':
        eval_dataset = ArxivMultiLabelDataset(
            csv_path=config['path'],
            tokenizer=config['tokenizer'],
            label_to_id=config['label_to_id']
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=config['batch_size'], num_workers=config.get('num_workers', 0), shuffle=False
        )
        return {'union_eval': eval_loader}

    print(f"Loading and splitting data for '{config['name']}' from: {config['path']}")
    
    full_df = pd.read_csv(config['path']).dropna(subset=['Abstracts', 'categories'])

    validation_split_size = config.get('validation_split', 0.1)
    train_df, val_df = train_test_split(
        full_df,
        test_size=validation_split_size,
        random_state=42 
    )
    
    print(f"  - Original size for this task: {len(full_df)}")
    print(f"  - Training on: {len(train_df)} samples")
    print(f"  - Validating on: {len(val_df)} samples")

    temp_dir = "./temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    train_path_temp = os.path.join(temp_dir, f"temp_train_split_{config['name']}.csv")
    val_path_temp = os.path.join(temp_dir, f"temp_val_split_{config['name']}.csv")
    train_df.to_csv(train_path_temp, index=False)
    val_df.to_csv(val_path_temp, index=False)

    train_dataset = ArxivMultiLabelDataset(
        csv_path=train_path_temp,
        tokenizer=config['tokenizer'],
        label_to_id=config['label_to_id'],
        label_subset=config.get('label_subset')
    )
    
    val_dataset = ArxivMultiLabelDataset(
        csv_path=val_path_temp,
        tokenizer=config['tokenizer'],
        label_to_id=config['label_to_id'],
        label_subset=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0),
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 0),
        shuffle=False
    )
    
    return {'train': train_loader, 'test': val_loader}