#!/usr/bin/env python3
"""
Gemma Fine-tuning Script
Fine-tune Google Gemma models for product review summarization using the organized training framework.

Usage:
    python finetune_gemma.py [options]

Examples:
    python finetune_gemma.py                           # Use default settings
    python finetune_gemma.py --epochs 5               # Train for 5 epochs
    python finetune_gemma.py --output my-model        # Custom output directory
    python finetune_gemma.py --batch-size 2           # Larger batch size
    python finetune_gemma.py --test-only               # Only test existing model
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Load environment variables and suppress warnings
load_dotenv()
warnings.filterwarnings('ignore')

# Import the organized training modules
from src.finetuning import GemmaTrainer, TrainingConfig


def check_requirements():
    """Check if all requirements are met for fine-tuning."""
    print("Checking requirements...")

    issues = []

    # Check HuggingFace token
    if not os.getenv('HUGGINGFACE_TOKEN'):
        issues.append("HUGGINGFACE_TOKEN environment variable not set")
    else:
        print("- HuggingFace token found")

    # Check training data
    data_path = "data/synthetic/fine_tuning_prompt_response.json"
    if not os.path.exists(data_path):
        issues.append(f"Training data not found: {data_path}")
    else:
        print(f"- Training data found: {data_path}")

    # Check if PyTorch and device availability
    try:
        import torch  # pylint: disable=import-outside-toplevel
        if torch.backends.mps.is_available():
            print("- Apple Silicon MPS available")
        elif torch.cuda.is_available():
            print("- CUDA available")
        else:
            print("- Using CPU (training will be slower)")
    except ImportError:
        issues.append("PyTorch not available")

    if issues:
        print("\nRequirements check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("All requirements satisfied!")
    return True


def create_parser():
    """Create argument parser for fine-tuning options."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma models for product review summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune_gemma.py                     # Default settings
  python finetune_gemma.py --epochs 5         # Train for 5 epochs
  python finetune_gemma.py --lr 1e-4          # Custom learning rate
  python finetune_gemma.py --output my-model  # Custom output directory
  python finetune_gemma.py --test-only        # Only test existing model
        """
    )

    parser.add_argument(
        '--epochs', type=int, default=3,
        help='Number of training epochs (default: 3)'
    )

    parser.add_argument(
        '--lr', '--learning-rate', type=float, default=2e-4,
        help='Learning rate (default: 2e-4)'
    )

    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Training batch size (default: 1)'
    )

    parser.add_argument(
        '--max-length', type=int, default=512,
        help='Maximum sequence length (default: 512)'
    )

    parser.add_argument(
        '--output', default='./roboreviews-gemma-finetuned',
        help='Output directory for fine-tuned model (default: ./roboreviews-gemma-finetuned)'
    )

    parser.add_argument(
        '--data-path',
        default='data/synthetic/fine_tuning_prompt_response.json',
        help='Path to training data '
             '(default: data/synthetic/fine_tuning_prompt_response.json)'
    )

    parser.add_argument(
        '--test-only', action='store_true',
        help='Only test existing model, skip training'
    )

    return parser


def main():
    """Main function for Gemma fine-tuning."""
    parser = create_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("GEMMA FINE-TUNING FOR PRODUCT REVIEWS")
    print("=" * 60)

    # Check requirements
    if not check_requirements():
        print("\nPlease fix the requirements and try again.")
        sys.exit(1)

    # Create training configuration
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Create trainer
    trainer = GemmaTrainer(config)

    if args.test_only:
        print(f"\nTesting existing model in {args.output}...")
        success = trainer.test_model()
        if success:
            print("Model test completed successfully!")
        else:
            print("Model test failed!")
            sys.exit(1)
    else:
        print("\nStarting fine-tuning process...")
        results = trainer.run_full_pipeline()

        if 'error' not in results:
            print("\nFine-tuning completed successfully!")
            print(f"- Final loss: {results['final_loss']:.4f}")
            print(f"- Training examples: {results['training_examples']}")
            print(f"- Model saved to: {results['output_dir']}")
            print(f"- Test result: {'PASSED' if results['test_success'] else 'FAILED'}")
        else:
            print(f"\nFine-tuning failed: {results['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
