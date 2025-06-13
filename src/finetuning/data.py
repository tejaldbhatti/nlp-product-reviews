"""
Data loading and processing for Gemma fine-tuning.
"""

import json
import os
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load the synthetic training data from fine_tuning_prompt_response.json"""
    print(f"Loading training data from {data_path}...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        training_examples = json.load(f)

    print(f"✓ Loaded {len(training_examples)} training examples")

    # Display sample to verify format
    if training_examples:
        sample = training_examples[0]
        print("\nSample training example:")
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Response: {sample['response'][:200]}...")
        print(f"Category: {sample['category']}")

    return training_examples


def format_for_gemma(instruction: str, response: str) -> str:
    """Format training examples using Gemma's chat template (matching existing model)"""
    return (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n{response}<end_of_turn>\n"
    )


class GemmaDataset(Dataset):
    """Dataset class for Gemma fine-tuning with proper formatting"""

    def __init__(self, training_examples: List[Dict[str, Any]],
                 tokenizer: AutoTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Format all examples with Gemma chat template
        self.formatted_texts = []
        for example in training_examples:
            formatted_text = format_for_gemma(
                example['instruction'],
                example['response']
            )
            self.formatted_texts.append(formatted_text)

        print(f"✓ Formatted {len(self.formatted_texts)} examples for Gemma")

        # Display sample formatted text
        if self.formatted_texts:
            print("\nSample Gemma format:")
            print(self.formatted_texts[0][:300] + "...")

    def __len__(self) -> int:
        return len(self.formatted_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.formatted_texts[idx]

        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM, labels = input_ids
        }

    def get_sample_formatted_text(self, idx: int = 0) -> str:
        """Get a sample formatted text for inspection"""
        if idx < len(self.formatted_texts):
            return self.formatted_texts[idx]
        return ""

    def validate_data(self) -> bool:
        """Validate the dataset integrity"""
        try:
            # Check if we can access first item
            if len(self) > 0:
                sample = self[0]
                required_keys = ['input_ids', 'attention_mask', 'labels']

                for key in required_keys:
                    if key not in sample:
                        print(f"Missing key: {key}")
                        return False

                    if not isinstance(sample[key], torch.Tensor):
                        print(f"{key} is not a tensor")
                        return False

                print("Dataset validation passed")
                return True

            print("Dataset is empty")
            return False

        except (RuntimeError, ValueError, KeyError) as e:
            print(f"Dataset validation failed: {e}")
            return False
