"""
Review Summarization Models
Handles local LLM integration for generating product review summaries and articles
"""

import logging
import os
import re
from typing import Dict

import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.pipelines import pipeline

try:
    from transformers import Gemma3ForCausalLM
except ImportError:
    Gemma3ForCausalLM = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    import google.generativeai as genai
    from google.generativeai import configure, GenerativeModel
except ImportError:
    genai = None
    configure = None
    GenerativeModel = None



# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def create_model_pipeline(model_type: str):
    """Factory function to create the appropriate model pipeline"""

    if model_type.lower() == "gemini-pro-flash":
        return create_gemini_pipeline()

    hf_token = os.getenv('HUGGINGFACE_TOKEN')

    # Model configurations - core models only
    model_configs = {
        "gemma-2b": {
            "path": "google/gemma-2-2b-it",
            "template": lambda prompt: (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            ),
            "extract_key": "<start_of_turn>model",
            "max_tokens": 384,
            "quantization": False
        },
        "gemma-3": {
            "path": "google/gemma-3-1b-it",  # Use instruction-tuned version
            "template": lambda prompt: (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            ),
            "extract_key": "<start_of_turn>model",
            "max_tokens": 400,
            "quantization": False,
            "model_class": "gemma3"  # Special flag for Gemma-3 handling
        },
        "mistral": {
            "path": "mistralai/Mistral-7B-Instruct-v0.2",
            "template": lambda prompt: f"<s>[INST] {prompt} [/INST]",
            "extract_key": "[/INST]",
            "max_tokens": 1024,
            "quantization": False
        },
        "qwen": {
            "path": "Qwen/Qwen2-7B-Instruct",
            "template": lambda prompt: (
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ),
            "extract_key": "<|im_start|>assistant",
            "max_tokens": 1024,
            "quantization": False
        },
        "qwen-finetuned": {
            "path": "./roboreviews-qwen-finetuned",
            "base_model": "Qwen/Qwen2-7B-Instruct",
            "template": lambda prompt: (
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ),
            "extract_key": "<|im_start|>assistant",
            "max_tokens": 512,
            "quantization": False,
            "is_finetuned": True
        },
        "gemma-3-finetuned": {
            "path": "./roboreviews-gemma-simple",
            "base_model": "google/gemma-3-1b-it",
            "template": lambda prompt: (
                f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            ),
            "extract_key": "<start_of_turn>model",
            "max_tokens": 600,  # Increased to prevent cutoffs
            "quantization": False,
            "is_finetuned": True,
            "model_class": "gemma3"  # Special flag for Gemma-3 handling
        },
        "gemini-pro-flash": {
            "type": "api",
            "provider": "google",
            "model_name": "gemini-2.5-flash-preview-04-17",
            "max_tokens": 512,
            "api_key_env": "GOOGLE_API_KEY"
        }
    }

    if model_type.lower() not in model_configs:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Choose from: {', '.join(model_configs.keys())}")

    config = model_configs[model_type.lower()]

    # Handle API-only models
    if config.get("type") == "api":
        if config.get("provider") == "google":
            return create_gemini_pipeline()
        raise ValueError(f"Unsupported API provider: {config.get('provider')}")

    return _create_local_model_pipeline(config, hf_token, model_type)


def _create_local_model_pipeline(config, hf_token, model_type):
    """Create pipeline for local models"""
    tokenizer = AutoTokenizer.from_pretrained(config["path"], token=hf_token)
    use_mps = torch.backends.mps.is_available()

    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "token": hf_token
    }

    if not use_mps:
        model_kwargs["device_map"] = "auto"

    if config["quantization"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = _load_model(config, model_kwargs, use_mps)
    pipe_kwargs = _prepare_pipeline_kwargs(config, model, tokenizer, model_type, use_mps)

    pipe = pipeline("text-generation", **pipe_kwargs)
    return pipe, config


def _load_model(config, model_kwargs, use_mps):
    """Load model based on configuration"""
    if config.get("is_finetuned", False):
        if PeftModel is None:
            raise ImportError("peft package not found. Install with: pip install peft")

        base_model_path = config["base_model"]
        # Handle Gemma-3 models specifically
        if config.get("model_class") == "gemma3":
            if Gemma3ForCausalLM is None:
                raise ImportError("Gemma3ForCausalLM not available. Please update transformers.")
            # Set MPS environment for Gemma-3 stability
            if use_mps:
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            base_model = Gemma3ForCausalLM.from_pretrained(
                base_model_path, **model_kwargs)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, **model_kwargs)

        if use_mps:
            base_model = base_model.to("mps")

        return PeftModel.from_pretrained(base_model, config["path"])

    # For non-finetuned models, check if it's Gemma-3
    if config.get("model_class") == "gemma3":
        if Gemma3ForCausalLM is None:
            raise ImportError("Gemma3ForCausalLM not available. Please update transformers.")

        if use_mps:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        model = Gemma3ForCausalLM.from_pretrained(
            config["path"], **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["path"], **model_kwargs)

    if use_mps:
        model = model.to("mps")

    return model


def _prepare_pipeline_kwargs(config, model, tokenizer, model_type, use_mps):
    """Prepare pipeline kwargs based on model type"""
    pipe_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": config["max_tokens"],
        "temperature": 0.9,
        "do_sample": True,
        "top_p": 0.85,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3
    }

    model_type_lower = model_type.lower()

    if model_type_lower == "gemma-2b":
        _configure_gemma_2b(pipe_kwargs, tokenizer)
    elif model_type_lower in ["qwen", "qwen-finetuned"]:
        _configure_qwen(pipe_kwargs, tokenizer, use_mps)
    elif model_type_lower in ["gemma-3", "gemma-3-finetuned"]:
        _configure_gemma_3(pipe_kwargs, tokenizer, model_type_lower, use_mps)

    return pipe_kwargs


def _configure_gemma_2b(pipe_kwargs, tokenizer):
    """Configure pipeline kwargs for Gemma 2B"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipe_kwargs["pad_token_id"] = tokenizer.eos_token_id
    pipe_kwargs["eos_token_id"] = tokenizer.eos_token_id
    pipe_kwargs["repetition_penalty"] = 1.15


def _configure_qwen(pipe_kwargs, tokenizer, use_mps):
    """Configure pipeline kwargs for Qwen models"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipe_kwargs["pad_token_id"] = tokenizer.eos_token_id
    pipe_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if use_mps:
        pipe_kwargs["device"] = "mps"


def _configure_gemma_3(pipe_kwargs, tokenizer, model_type_lower, use_mps):
    """Configure pipeline kwargs for Gemma 3 models"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pipe_kwargs["pad_token_id"] = tokenizer.eos_token_id
    pipe_kwargs["eos_token_id"] = tokenizer.eos_token_id

    # Different settings for fine-tuned vs base models
    if model_type_lower == "gemma-3-finetuned":
        _configure_finetuned_gemma_3(pipe_kwargs, tokenizer)
    else:
        pipe_kwargs["repetition_penalty"] = 1.1

    # Conservative settings for MPS stability
    if use_mps:
        pipe_kwargs["device"] = "mps"
        pipe_kwargs["do_sample"] = False  # Use greedy decoding for stability
        pipe_kwargs["temperature"] = None  # Disable sampling parameters
        pipe_kwargs["top_p"] = None
        pipe_kwargs["top_k"] = None


def _configure_finetuned_gemma_3(pipe_kwargs, tokenizer):
    """Configure pipeline kwargs specifically for fine-tuned Gemma 3"""
    # Optimized settings for fine-tuned model
    pipe_kwargs["temperature"] = 0.7  # Lower for more focused output
    pipe_kwargs["repetition_penalty"] = 1.1  # Reduced to allow natural repetition
    pipe_kwargs["top_p"] = 0.9  # Slightly higher for better quality
    pipe_kwargs["max_new_tokens"] = 600  # Increased to prevent cutoffs

    # Add EOS tokens to prevent content bleeding
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        try:
            end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_turn_id != tokenizer.unk_token_id:
                pipe_kwargs["eos_token_id"] = [tokenizer.eos_token_id, end_turn_id]
        except (AttributeError, KeyError, ValueError):
            pass  # Fall back to default EOS token

    pipe_kwargs["max_new_tokens"] = 500
    # Prevent repetitive phrases
    pipe_kwargs["no_repeat_ngram_size"] = 3


def generate_text(pipe_or_model, template_fn, extract_key: str,
                  prompt: str, model_config: Dict = None) -> str:
    """Generate text using the model pipeline or API"""
    try:
        if model_config and model_config.get("type") == "api":
            if model_config.get("provider") == "google":
                response = generate_gemini_text(pipe_or_model, prompt)
            else:
                provider = model_config.get('provider')
                raise ValueError(f"Unsupported API provider: {provider}")
        else:
            formatted_prompt = template_fn(prompt)
            # Special handling for Gemma-3 models on MPS
            if (model_config and model_config.get("model_class") == "gemma3" and
                    torch.backends.mps.is_available()):
                # Use conservative settings for all Gemma-3 models on MPS
                # Fine-tuned models will rely on their learned patterns with greedy decoding
                generation_kwargs = {
                    "max_new_tokens": model_config.get("max_tokens", 400) if model_config else 400,
                    "do_sample": False,  # Greedy decoding for MPS stability
                    "pad_token_id": pipe_or_model.tokenizer.eos_token_id,
                    "eos_token_id": pipe_or_model.tokenizer.eos_token_id,
                    "repetition_penalty": 1.05 if (model_config and model_config.get("is_finetuned")) else 1.1,
                    "return_full_text": True,
                }
                result = pipe_or_model(formatted_prompt, **generation_kwargs)
            else:
                # Standard generation for other models
                result = pipe_or_model(formatted_prompt)
            generated_text = result[0]["generated_text"]

            if extract_key in generated_text:
                response = generated_text.split(extract_key)[-1]
            else:
                response = generated_text

        response = response.replace('\r\n', '\n').replace('\r', '\n')
        return response.strip()

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error generating text: %s", str(e))
        return "Error generating content for this section."



def clean_recommendation_text(text: str) -> str:
    """Enhanced cleaning of LLM output to remove conversational prefixes and training artifacts"""

    # Remove obvious conversational prefixes
    conversational_patterns = [
        r"^Okay,?\s*here'?s?\s*",
        r"^Here'?s?\s*",
    ]

    for pattern in conversational_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Remove training artifacts and garbage text patterns (more targeted)
    artifact_patterns = [
        r'\n\n+user\b.*$',              # Remove "user" conversations
        r'\n\n+[a-z]+\n\n+user.*$',    # Remove name+user patterns like "atman\nuser"
        r'<start_of_turn>.*$',         # Remove turn markers
        r'<end_of_turn>.*$',           # Remove turn markers
        r'\n\n+###\s*\*\*Beyond.*$',   # Remove incomplete sections starting with "Beyond"
        r'[\u4e00-\u9fff]+.*$',        # Remove Chinese characters and everything after
        r'[\u0900-\u097f]+.*$',        # Remove Hindi/Devanagari and everything after
        r'[\u0980-\u09ff]+.*$',        # Remove Bengali and everything after
        r'[\u0b80-\u0bff]+.*$',        # Remove Tamil and everything after
        r'[\u10a0-\u10ff]+.*$',        # Remove Georgian and everything after
        r'[\u0400-\u04ff]+.*$',        # Remove Cyrillic characters and everything after
        r'<unused\d+>.*$',             # Remove unused tokens
        r'[\ud800-\udfff]+.*$',        # Remove Unicode surrogates/garbage
        r'[🐂🗕📝🎯⚡️💡🔥]+.*$',       # Remove specific emoji garbage
        r'\s+[🐂🗕📝🎯⚡️💡🔥]+.*$',   # Remove emoji garbage with whitespace
        r'\n\n\n+.*$',                # Remove trailing content after triple newlines
        r'[\ud800-\udfff].*$',         # More specific Unicode surrogate removal
        r'\n\n+How would you approach.*$',  # Remove training questions
        r'\n\n+abhishek.*$',           # Remove specific training artifacts
        r'\n\n+atman.*$',              # Remove specific training artifacts
        r'\s+всему.*$',                # Remove specific Cyrillic artifacts
    ]
    for pattern in artifact_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)

    return text.strip()


def create_gemini_pipeline():
    """Create Gemini API pipeline"""
    if genai is None:
        raise ImportError("google-generativeai package not found. "
                         "Install with: pip install google-generativeai")

    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found")

    # Configure the API
    if configure is None:
        raise ImportError("google.generativeai not available")
    configure(api_key=api_key)

    # Create the model
    if GenerativeModel is None:
        raise ImportError("google.generativeai not available")
    model = GenerativeModel('gemini-2.5-flash-preview-04-17')

    # Return model and config
    config = {
        "type": "api",
        "provider": "google",
        "model_name": "gemini-2.5-flash-preview-04-17",
        "max_tokens": 512,
        "template": lambda prompt: prompt,  # Gemini doesn't need special formatting
        "extract_key": "",  # No extraction needed for API responses
    }

    return model, config


def generate_gemini_text(model, prompt: str) -> str:
    """Generate text using Gemini API"""
    try:
        response = model.generate_content(prompt)

        if response.text:
            return response.text
        return "Error: No content generated by Gemini API"

    except (ConnectionError, TimeoutError, ValueError) as exc:
        logger.error("Error generating text with Gemini API: %s", str(exc))
        return "Error generating content with Gemini API."
