"""
Review Summarization Models
Handles local LLM integration for generating product review summaries and articles
"""

import json
import logging
import os
import re
import time
from typing import List, Dict

import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
from .prompts import few_shot_comparison_prompt

try:
    import google.generativeai as genai
except ImportError:
    genai = None



# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def create_model_pipeline(model_type: str):
    """Factory function to create the appropriate model pipeline"""

    # Check if this is an API-based model
    if model_type.lower() == "gemini-pro-flash":
        return create_gemini_pipeline()

    # Get HuggingFace token from environment
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["path"], token=hf_token)

    # Check for MPS availability (Apple Silicon)
    use_mps = torch.backends.mps.is_available()
    print(f"MPS available: {use_mps}")

    # Configure model loading (avoid device_map="auto" for MPS)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "token": hf_token
    }

    # Only use device_map for non-MPS systems
    if not use_mps:
        model_kwargs["device_map"] = "auto"
        print("Using device_map='auto' for non-MPS system")
    else:
        print("Using explicit MPS device placement (avoiding device_map='auto')")

    # Add quantization if specified
    if config["quantization"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    # No special model-specific settings needed for current models

    # Load model (handle fine-tuned models)
    if config.get("is_finetuned", False):
        # Load LoRA fine-tuned model using PEFT
        if PeftModel is None:
            raise ImportError("peft package not found. Install with: pip install peft")

        # Load base model first
        base_model_path = config["base_model"]
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, **model_kwargs)

        # Move base model to MPS if available (before applying adapter)
        if use_mps:
            base_model = base_model.to("mps")
            print("Moved base model to MPS device")

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, config["path"])
        print(f"Loaded fine-tuned model from {config['path']}")
    else:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            config["path"], **model_kwargs)

        # Move to MPS if available
        if use_mps:
            model = model.to("mps")
            print("Moved model to MPS device")

    # Create pipeline with creative parameters
    pipe_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": config["max_tokens"],
        "temperature": 0.9,  # Increased for more creativity
        "do_sample": True,
        "top_p": 0.85,       # Slightly lower for quality control
        "top_k": 50,         # Add diversity in word selection
        "repetition_penalty": 1.2,  # Reduce repetitive phrases
        "no_repeat_ngram_size": 3    # Prevent repetitive 3-word sequences
    }

    # Add model-specific configurations
    if model_type.lower() == "gemma-2b":
        # Fix tokenizer and generation issues for Gemma models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pipe_kwargs["pad_token_id"] = tokenizer.eos_token_id
        pipe_kwargs["eos_token_id"] = tokenizer.eos_token_id
        # Keep the creative parameters but ensure stability
        pipe_kwargs["repetition_penalty"] = 1.15  # Slightly more conservative for Gemma
    elif model_type.lower() in ["qwen", "qwen-finetuned"]:
        # Fix tokenizer issues for Qwen models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pipe_kwargs["pad_token_id"] = tokenizer.eos_token_id
        pipe_kwargs["eos_token_id"] = tokenizer.eos_token_id
        # Ensure MPS device is used in pipeline
        if use_mps:
            pipe_kwargs["device"] = "mps"
            print("Explicitly set pipeline device to MPS for Qwen model")

    pipe = pipeline("text-generation", **pipe_kwargs)

    print(f"Loaded {config['path']} successfully")

    return pipe, config


def generate_text(pipe_or_model, template_fn, extract_key: str,
                  prompt: str, model_config: dict = None) -> str:
    """Generate text using the model pipeline or API"""
    try:
        print(f"    - Starting text generation "
              f"(prompt length: {len(prompt)} chars)...")
        start_time = time.time()

        # Check if this is an API model
        if model_config and model_config.get("type") == "api":
            # Use API generation
            if model_config.get("provider") == "google":
                response = generate_gemini_text(pipe_or_model, prompt)
            else:
                raise ValueError(
                    f"Unsupported API provider: {model_config.get('provider')}")
        else:
            # Use local model pipeline
            # Format prompt using model-specific template
            formatted_prompt = template_fn(prompt)

            # Generate text
            result = pipe_or_model(formatted_prompt)
            generated_text = result[0]["generated_text"]

            # Extract only the generated part
            if extract_key in generated_text:
                response = generated_text.split(extract_key)[-1].strip()
            else:
                response = generated_text.strip()

        inference_time = time.time() - start_time
        print(f"    - Text generation completed "
              f"({len(response)} chars in {inference_time:.2f}s)")
        return response

    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error generating text: %s", str(e))
        print("    - ERROR in text generation:", str(e))
        return "Error generating content for this section."


def generate_comparison_article(
        model_pipeline,
        model_config: Dict,
        products: List[Dict],
        category: str,
        sample_reviews: Dict = None) -> str:
    """Generate buying recommendation text for products in category using pre-loaded model"""
    # Generate prompt with sample reviews
    prompt = few_shot_comparison_prompt(products, category, sample_reviews)

    # Generate text using pre-loaded pipeline or API
    raw_text = generate_text(
        model_pipeline,
        model_config["template"],
        model_config["extract_key"],
        prompt,
        model_config)

    # Clean up conversational responses (no JSON parsing needed now)
    cleaned_text = clean_recommendation_text(raw_text)
    return cleaned_text


def clean_recommendation_text(text: str) -> str:
    """Clean structured markdown recommendation text"""

    # Remove conversational patterns that might appear before the structured content
    conversational_patterns = [
        r"^Okay,?\s*here'?s?\s+.*?[:\n]",
        r"^Here'?s?\s+.*?[:\n]",
        r"^Based on.*?analysis,?\s*",
        r"^After analyzing.*?reviews,?\s*",
        r"^Task:\s*.*?\n",
        r"^Create.*?guide.*?\n",
        r"Generated by.*?in.*?seconds?.*$",
        r"Would you like me to.*$",
        r"Let me know.*$",
    ]

    for pattern in conversational_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Clean up markdown formatting issues
    text = re.sub(r'#{4,}', '###', text)  # Max 3 levels of headers
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive newlines
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading spaces

    # Remove any remaining JSON artifacts
    text = re.sub(r'[{}"\[\]]', '', text)

    # Clean up and preserve markdown structure
    text = text.strip()

    return text


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
    genai.configure(api_key=api_key)

    # Create the model
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

    # Return model and config
    config = {
        "type": "api",
        "provider": "google",
        "model_name": "gemini-2.5-flash-preview-04-17",
        "max_tokens": 512,
        "template": lambda prompt: prompt,  # Gemini doesn't need special formatting
        "extract_key": "",  # No extraction needed for API responses
    }

    print("Loaded Gemini 2.5 Flash Preview API successfully")
    return model, config


def generate_gemini_text(model, prompt: str) -> str:
    """Generate text using Gemini API"""
    try:
        print(f"    - Starting Gemini API generation "
              f"(prompt length: {len(prompt)} chars)...")
        start_time = time.time()

        response = model.generate_content(prompt)

        if response.text:
            generated_text = response.text.strip()
        else:
            generated_text = "Error: No content generated by Gemini API"

        inference_time = time.time() - start_time
        print(f"    - Gemini API generation completed "
              f"({len(generated_text)} chars in {inference_time:.2f}s)")
        return generated_text

    except (ConnectionError, TimeoutError, ValueError) as exc:
        logger.error("Error generating text with Gemini API: %s", str(exc))
        print(f"    - ERROR in Gemini API generation: {str(exc)}")
        return "Error generating content with Gemini API."


def clean_generated_text(text: str) -> str:
    """Remove conversational responses, markdown syntax, and clean up generated text"""

    # Try to extract and parse JSON if the text looks like JSON
    if '{' in text and '}' in text:
        try:
            # Remove markdown code blocks first
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = re.sub(r'`json\s*', '', text)
            text = re.sub(r'`\s*', '', text)

            # Extract JSON-like content - find the first complete JSON object
            start = text.find('{')
            if start != -1:
                brace_count = 0
                end = start
                for i, char in enumerate(text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break

                json_text = text[start:end]
                # Clean up common JSON formatting issues
                json_text = re.sub(r'\.pros":', '"pros":', json_text)
                json_text = re.sub(r'\.cons":', '"cons":', json_text)
                json_text = re.sub(r'\.recommendation":', '"recommendation":', json_text)
                json_text = re.sub(r'\.rating":', '"rating":', json_text)
                json_text = re.sub(r'\.total_reviews":', '"total_reviews":', json_text)

                # Fix all quote types and casing issues
                quote_pattern = r'[""''"'']'
                json_text = re.sub(quote_pattern, '"', json_text)
                json_text = re.sub(r'"Title":', '"title":', json_text)
                json_text = re.sub(r'"Pros":', '"pros":', json_text)
                json_text = re.sub(r'"Cons?":', '"cons":', json_text)  # Handle "con" and "cons"
                json_text = re.sub(r'"Recommendation":', '"recommendation":', json_text)
                json_text = re.sub(r'"Rating":', '"rating":', json_text)
                json_text = re.sub(r'"Total[_ ]?[Rr]eviews?":', '"total_reviews":', json_text)

                # Fix field name variations
                json_text = re.sub(r'"con":', '"cons":', json_text)
                json_text = re.sub(r'"total_review[s]?":', '"total_reviews":', json_text)
                json_text = re.sub(r'"total_recipes":', '"total_reviews":', json_text)  # Fix typo
                json_text = re.sub(r'"total_users":', '"total_reviews":', json_text)  # Fix typo

                # Clean up malformed arrays and extra characters
                json_text = re.sub(r',\s*""', '', json_text)
                json_text = re.sub(r'"\s*",\s*""', '"', json_text)
                json_text = re.sub(r'，', ',', json_text)  # Fix Chinese comma
                json_text = re.sub(r'：', ':', json_text)  # Fix Chinese colon
                json_text = re.sub(r'、', ',', json_text)  # Fix Japanese comma
                json_text = re.sub(r'\\\\', r'\\', json_text)  # Fix double escapes

                # Remove standalone field names that got mixed into content
                json_text = re.sub(r',\s*"rating"\s*,', ',', json_text)
                json_text = re.sub(r',\s*"rating"\s*\]', ']', json_text)
                json_text = re.sub(r'\[\s*"rating"\s*,', '[', json_text)
                json_text = re.sub(r'\[\s*"rating"\s*\]', '[]', json_text)
                json_text = re.sub(r'"rating"\s*:', '"rating":', json_text)
                json_text = re.sub(r'"rating"\s*[^:,}\]]+', '"rating"', json_text)
                json_text = re.sub(r',\s*"rating"\s*[^:,}\]]+', '', json_text)

                # Remove trailing characters and fix malformed endings
                json_text = re.sub(r'\}\s*\]', '}', json_text)  # Remove trailing ]
                json_text = re.sub(r',\s*\}', '}', json_text)  # Remove trailing comma before }
                json_text = re.sub(r',\s*\]', ']', json_text)  # Remove trailing comma before ]

                # Ensure proper array format for cons
                json_text = re.sub(r'"cons":\s*"([^"]+)"', r'"cons": ["\1"]', json_text)

                # Try to parse as JSON
                parsed = json.loads(json_text)
                return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, AttributeError, ValueError):
            # If JSON parsing fails, fall back to text cleaning
            pass
    # Remove common conversational openings and repetitive phrases
    conversational_patterns = [
        r"^Okay,?\s*here'?s?\s+.*?[:\n]",
        r"^Here'?s?\s+.*?[:\n]",
        r"^Let's be honest.*?[,\.\n]",
        r"^Let\u2019s be honest.*?[,\.\n]",  # Smart quote version
        r"^.*incorporating customer review analysis.*?[:\n]",
        r"^.*tech blog.*?post.*?[:\n]",
        r"^.*buying guide.*?crafted.*?[:\n]",
        r"^.*mirroring the format.*?[:\n]",
        r"^After analyzing.*?reviews,?\s*",
        r"^Based on.*?reviews,?\s*",
        r"Generated by.*?in.*?seconds?.*$",
        r"Would you like me to.*$",
        r"Let me know.*$",
        r"Given your prompt.*$",
    ]

    for pattern in conversational_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Remove markdown syntax more aggressively
    markdown_patterns = [
        (r"^#{1,6}\s*(.*)$", r"\1"),      # ## Header -> Header
        (r"\*\*(.*?)\*\*", r"\1"),        # **bold** -> bold
        (r"\*(.*?)\*", r"\1"),            # *italic* -> italic
        (r"`(.*?)`", r"\1"),              # `code` -> code
        (r"^\s*[\*\-\+]\s*", ""),         # Remove bullet points
        (r"^\s*\d+\.\s*", ""),            # Remove numbered lists
        (r"^\s*>\s*", ""),                # Remove blockquotes
        (r"\[([^\]]+)\]\([^\)]+\)", r"\1"), # [text](link) -> text
    ]

    for pattern, replacement in markdown_patterns:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # Remove attribution lines and meta text
    meta_patterns = [
        r"--+\s*$",                       # Remove divider lines
        r"^\s*\*{3,}\s*$",               # Remove *** lines
        r"^\s*={3,}\s*$",                # Remove === lines
        r"Generated with.*$",
        r"Model used:.*$",
        r"Processing time:.*$",
    ]

    for pattern in meta_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Clean up extra whitespace and newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'^\s+|\s+$', '', text)          # Trim whitespace
    text = re.sub(r' +', ' ', text)                # Multiple spaces to single space

    return text
