"""
Utility functions for Gemma fine-tuning.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def setup_environment() -> str:
    """Setup environment for Apple Silicon MPS optimization"""
    print("Setting up environment for Apple Silicon...")

    # Critical MPS environment variables (from finetuning_guide.md)
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ Apple Silicon MPS available")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✓ CUDA available")
    else:
        device = "cpu"
        print("⚠ Using CPU (slower training)")

    return device


def load_gemma_model(model_name: str, device: str, torch_dtype: str = "float16"):
    """Load Gemma model and tokenizer with MPS optimizations"""
    print(f"Loading {model_name} model and tokenizer...")

    # Get HuggingFace token
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")

    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.float16)

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )

    # Gemma-specific tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("✓ Tokenizer loaded and configured")

    # Load the model directly to device
    print(f"Loading model onto {device} device...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype_obj,
        trust_remote_code=True,
        token=hf_token
    )
    model = model.to(device)

    print(f"✓ Model loaded on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    return model, tokenizer


def test_model(output_dir: str, device: str, model_name: str = "google/gemma-3-1b-it"):
    """Test the fine-tuned model with a sample prompt"""
    print(f"\nTesting fine-tuned model from {output_dir}...")

    # Check if model directory exists
    if not os.path.exists(output_dir):
        print(f"Model directory not found: {output_dir}")
        return False

    # Check if adapter config exists
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"No fine-tuned model found in {output_dir}")
        print("Please train a model first or specify the correct output directory")
        return False

    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            token=os.getenv('HUGGINGFACE_TOKEN')
        )

        # Load fine-tuned adapters
        model = PeftModel.from_pretrained(base_model, output_dir)
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(output_dir)

        # Test prompt
        test_instruction = ("Create a product recommendation guide for Smart Watches "
                          "based on customer reviews and ratings.")
        formatted_prompt = (f"<start_of_turn>user\n{test_instruction}<end_of_turn>\n"
                          f"<start_of_turn>model\n")

        print(f"Test prompt: {test_instruction}")

        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<end_of_turn>")
                ]
            )

        # Decode and display
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = (full_response.find("<start_of_turn>model\n") +
                         len("<start_of_turn>model\n"))
        generated_response = full_response[response_start:].strip()

        if "<end_of_turn>" in generated_response:
            generated_response = generated_response.split("<end_of_turn>")[0].strip()

        print("\nGenerated response:")
        print("-" * 40)
        print(generated_response)
        print("-" * 40)

        return True

    except (OSError, ValueError, RuntimeError) as e:
        print(f"Error testing model: {e}")
        return False


def clear_gpu_cache(device: str):
    """Clear GPU cache based on device type"""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def get_model_info(model) -> dict:
    """Get information about the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0
    }
