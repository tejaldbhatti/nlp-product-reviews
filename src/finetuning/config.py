"""
Training configuration and hyperparameters for Gemma fine-tuning.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from peft import LoraConfig, TaskType


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    model_name: str = "google/gemma-3-1b-it"
    torch_dtype: str = "float16"
    device: Optional[str] = None  # Will be auto-detected


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration"""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Configuration class for Gemma fine-tuning training"""

    # Basic Configuration
    output_dir: str = "./roboreviews-gemma-finetuned"
    data_path: str = "data/synthetic/fine_tuning_prompt_response.json"

    # Training Parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    max_length: int = 512
    weight_decay: float = 0.01

    # Nested configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate data path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data not found: {self.data_path}")

    def get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration for PEFT"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules
        )

    @property
    def model_name(self) -> str:
        """Get model name"""
        return self.model_config.model_name

    @property
    def device(self) -> Optional[str]:
        """Get device"""
        return self.model_config.device

    @property
    def torch_dtype(self) -> str:
        """Get torch dtype"""
        return self.model_config.torch_dtype

    @property
    def num_workers(self) -> int:
        """DataLoader num_workers (MPS optimized)"""
        return 0  # Critical: must be 0 for MPS

    @property
    def pin_memory(self) -> bool:
        """DataLoader pin_memory (MPS optimized)"""
        return False  # Critical: disable for MPS

    @property
    def drop_last(self) -> bool:
        """DataLoader drop_last setting"""
        return False

    @property
    def log_steps(self) -> int:
        """Logging frequency"""
        return 5

    @property
    def cache_clear_steps(self) -> int:
        """MPS cache clearing frequency"""
        return 10

    @property
    def max_new_tokens(self) -> int:
        """Generation max tokens"""
        return 300

    @property
    def temperature(self) -> float:
        """Generation temperature"""
        return 0.7

    @property
    def do_sample(self) -> bool:
        """Generation sampling setting"""
        return True

    def get_training_summary(self) -> str:
        """Get a summary of training configuration"""
        return f"""
Training Configuration Summary:
{'='*40}
Model: {self.model_name}
Output: {self.output_dir}
Data: {self.data_path}

Training Parameters:
- Epochs: {self.num_epochs}
- Learning Rate: {self.learning_rate}
- Batch Size: {self.batch_size}
- Max Length: {self.max_length}

LoRA Parameters:
- Rank (r): {self.lora_config.r}
- Alpha: {self.lora_config.alpha}
- Dropout: {self.lora_config.dropout}
- Target Modules: {', '.join(self.lora_config.target_modules)}
{'='*40}
        """.strip()
