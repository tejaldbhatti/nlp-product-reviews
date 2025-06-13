"""
Core training logic for Gemma fine-tuning with MPS optimizations.
"""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import TrainingConfig
from .data import GemmaDataset, load_training_data
from .utils import setup_environment, load_gemma_model, clear_gpu_cache, get_model_info, test_model


class GemmaTrainer:
    """Main trainer class for Gemma fine-tuning"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device or setup_environment()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.dataset: Optional[GemmaDataset] = None
        self.dataloader: Optional[DataLoader] = None

    def setup_model(self):
        """Load and setup the model for training"""
        print("Setting up Gemma model for training...")

        # Load base model and tokenizer
        self.model, self.tokenizer = load_gemma_model(
            self.config.model_name,
            self.device,
            self.config.torch_dtype
        )

        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        lora_config = self.config.get_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # Print model info
        model_info = get_model_info(self.model)
        print(f"Trainable parameters: {model_info['trainable_parameters']:,} "
              f"({model_info['trainable_percentage']:.2f}%)")

        print("✓ Model setup completed")

    def setup_data(self):
        """Load and setup training data"""
        print("Setting up training data...")

        # Load training examples
        training_examples = load_training_data(self.config.data_path)

        # Ensure tokenizer is available
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")

        # Create dataset
        self.dataset = GemmaDataset(
            training_examples,
            self.tokenizer,
            max_length=self.config.max_length
        )

        # Validate dataset
        if not self.dataset.validate_data():
            raise ValueError("Dataset validation failed")

        # Create dataloader with MPS optimizations
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last
        )

        print("✓ Data setup completed")

    def train(self) -> float:
        """Main training function"""
        print("\n" + "="*60)
        print("STARTING GEMMA FINE-TUNING")
        print("="*60)
        print(self.config.get_training_summary())

        # Setup model and data
        self.setup_model()
        self.setup_data()

        # Ensure model and dataloader are available
        if self.model is None or self.dataloader is None:
            raise ValueError("Model or dataloader not initialized")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        print("\nTraining configuration:")
        print(f"- Total steps: {len(self.dataloader) * self.config.num_epochs}")
        print(f"- Device: {self.device}")

        # Training loop
        return self._training_loop(optimizer)

    def _training_loop(self, optimizer) -> float:
        """Execute the training loop with MPS optimizations"""
        if self.model is None or self.dataloader is None:
            raise ValueError("Model or dataloader not initialized")

        self.model.train()
        total_loss = 0
        step_count = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Training")):
                try:
                    # Move batch to device with non_blocking for MPS optimization
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Track loss
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    step_count += 1

                    # Clear cache periodically
                    if batch_idx % self.config.cache_clear_steps == 0:
                        clear_gpu_cache(self.device)

                    # Log progress
                    if step_count % self.config.log_steps == 0:
                        print(f"Step {step_count}, Loss: {loss.item():.4f}")

                except (RuntimeError, ValueError, OSError) as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    clear_gpu_cache(self.device)
                    continue

            dataloader_len = len(self.dataloader)
            avg_epoch_loss = epoch_loss / dataloader_len if dataloader_len > 0 else 0
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss / step_count if step_count > 0 else 0
        print("\nTraining completed!")
        print(f"Average loss: {avg_total_loss:.4f}")
        print(f"Total steps: {step_count}")

        # Save the model
        self._save_model()

        return avg_total_loss

    def _save_model(self):
        """Save the fine-tuned model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized")

        print(f"\nSaving model to {self.config.output_dir}...")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print("✓ Model saved successfully!")

    def test_model(self) -> bool:
        """Test the fine-tuned model"""
        return test_model(self.config.output_dir, self.device, self.config.model_name)

    def run_full_pipeline(self) -> dict:
        """Run the complete training and testing pipeline"""
        try:
            # Train the model
            final_loss = self.train()

            # Test the model
            test_success = self.test_model()

            # Print completion summary
            print("\n" + "="*60)
            print("GEMMA FINE-TUNING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Model fine-tuned with LoRA adapters")
            if self.dataset is not None:
                print(f"Training data: {len(self.dataset)} professional examples")
            print(f"Final loss: {final_loss:.4f}")
            print(f"Model saved to: {self.config.output_dir}")
            print(f"Test {'PASSED' if test_success else 'FAILED'}")
            print("Ready for integration into pipeline")

            print("\nTo use this model in your pipeline:")
            print("1. Update model path in src/summarization/models.py")
            print("2. Run: python run_pipeline.py gemma-3-finetuned")

            return {
                'final_loss': final_loss,
                'test_success': test_success,
                'output_dir': self.config.output_dir,
                'training_examples': len(self.dataset) if self.dataset is not None else 0
            }

        except (RuntimeError, ValueError, OSError) as e:
            print(f"Error during training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
