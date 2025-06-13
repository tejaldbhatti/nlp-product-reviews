"""
Simple CSV-Based Summarization Pipeline
Run this after the sentiment and category CSV files are ready.

Usage:
    python run_pipeline.py <model_name>
    
Available models:
    - gemma-2b: Google Gemma 2B (fast, general purpose)
    - gemma-3: Google Gemma 3-1B instruct (efficient, MPS optimized)
    - mistral: Mistral 7B (balanced performance)
    - qwen: Qwen 7B (high quality)
    - qwen-finetuned: Fine-tuned Qwen 7B (specialized)
    - gemma-3-finetuned: Fine-tuned Gemma 3-1B (specialized, efficient)
    - gemini-pro-flash: Google Gemini API (requires API key)

Examples:
    python run_pipeline.py gemma-2b
    python run_pipeline.py gemma-3
    python run_pipeline.py gemma-3-finetuned
    python run_pipeline.py qwen-finetuned
"""
import os
import sys
import math
import json
import argparse
from src.summarization.pipeline import SummarizationPipeline


# Available models with descriptions
AVAILABLE_MODELS = {
    "gemma-2b": {
        "name": "Google Gemma 2B",
        "description": "Fast, general purpose model (2B parameters)",
        "requirements": "HuggingFace token"
    },
    "gemma-3": {
        "name": "Google Gemma 3-1B (Instruct)",
        "description": "Efficient instruction-tuned model (1B parameters, MPS optimized)",
        "requirements": "HuggingFace token"
    },
    "mistral": {
        "name": "Mistral 7B",
        "description": "Balanced performance model (7B parameters)",
        "requirements": "HuggingFace token"
    },
    "qwen": {
        "name": "Qwen 7B", 
        "description": "High quality base model (7B parameters)",
        "requirements": "HuggingFace token"
    },
    "qwen-finetuned": {
        "name": "Fine-tuned Qwen 7B",
        "description": "Specialized for product recommendations (7B parameters)",
        "requirements": "HuggingFace token + ./roboreviews-qwen-finetuned/"
    },
    "gemma-3-finetuned": {
        "name": "Fine-tuned Gemma 3-1B",
        "description": "Specialized, efficient model (1B parameters, MPS optimized)",
        "requirements": "HuggingFace token + ./roboreviews-gemma-finetuned/"
    },
    "gemini-pro-flash": {
        "name": "Google Gemini Pro Flash",
        "description": "Cloud API model (very fast, high quality)",
        "requirements": "Google API key"
    }
}


def clean_data_recursively(obj):
    """Recursively clean data to handle NaN values"""
    if isinstance(obj, dict):
        return {key: clean_data_recursively(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_data_recursively(item) for item in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def json_serializer(obj):
    """Custom JSON serializer to handle NaN values"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return str(obj)

def print_available_models():
    """Print information about available models."""
    print("🤖 Available Models:")
    print("=" * 80)
    
    for model_key, model_info in AVAILABLE_MODELS.items():
        print(f"\n📦 {model_key}")
        print(f"   Name: {model_info['name']}")
        print(f"   Description: {model_info['description']}")
        print(f"   Requirements: {model_info['requirements']}")
    
    print("\n" + "=" * 80)
    print("\nUsage Examples:")
    print("   python run_pipeline.py gemma-2b")
    print("   python run_pipeline.py gemma-3")
    print("   python run_pipeline.py gemma-3-finetuned")
    print("   python run_pipeline.py qwen-finetuned")
    print("   python run_pipeline.py gemini-pro-flash")
    
    print("\n🔧 Setup Requirements:")
    print("   • HuggingFace token: Set HUGGINGFACE_TOKEN in .env")
    print("   • Google API key: Set GOOGLE_API_KEY in .env (for gemini-pro-flash)")
    print("   • Fine-tuned models: Run training scripts first")

def validate_model_availability(model_type):
    """Check if the specified model is available and ready to use."""
    print(f"Validating model: {model_type}")
    
    # Check fine-tuned models
    if model_type == "qwen-finetuned":
        model_path = "./roboreviews-qwen-finetuned"
        if not os.path.exists(model_path):
            print(f"Error: Fine-tuned model not found: {model_path}")
            print("   Please run fine-tuning first or use a different model")
            return False
        print(f"Fine-tuned model found: {model_path}")
    
    elif model_type == "gemma-3-finetuned":
        model_path = "./roboreviews-gemma-simple"
        if not os.path.exists(model_path):
            print(f"Error: Fine-tuned model not found: {model_path}")
            print("   Please run fine-tuning first or use a different model")
            return False
        print(f"Fine-tuned model found: {model_path}")
    
    # Check API requirements
    elif model_type == "gemini-pro-flash":
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv('GOOGLE_API_KEY'):
            print("Error: GOOGLE_API_KEY not found in environment")
            print("   Please set it in your .env file")
            return False
        print("Google API key found")
    
    # Check HuggingFace token for local models
    if model_type != "gemini-pro-flash":
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv('HUGGINGFACE_TOKEN'):
            print("Error: HUGGINGFACE_TOKEN not found in environment")
            print("   Please set it in your .env file")
            return False
        print("HuggingFace token found")
    
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the RoboReviews summarization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
{chr(10).join([f"  {key:20} - {info['description']}" for key, info in AVAILABLE_MODELS.items()])}

Examples:
  python run_pipeline.py gemma-2b
  python run_pipeline.py gemma-3-finetuned
  python run_pipeline.py qwen-finetuned
        """
    )
    
    parser.add_argument(
        "model", 
        nargs='?',  # Optional positional argument
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use for text generation"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--sentiment-csv",
        default="results/sentiment_results.csv",
        help="Path to sentiment analysis CSV file"
    )
    
    
    return parser.parse_args()

def main():
    """Main function to run the NLP product review summarization pipeline."""
    args = parse_arguments()
    
    # Handle list models request
    if args.list_models:
        print_available_models()
        return
    
    # Check if model was provided
    if not args.model:
        print("Error: Model name is required")
        print("\n" + "="*50)
        print_available_models()
        print("\n" + "="*50)
        print("\nPlease specify a model:")
        print("   python run_pipeline.py <model_name>")
        print("   python run_pipeline.py --list-models")
        sys.exit(1)
    
    model_type = args.model
    
    print("=== ROBOREVIEWS SUMMARIZATION PIPELINE ===")
    print(f"Model: {AVAILABLE_MODELS[model_type]['name']}")
    print(f"Description: {AVAILABLE_MODELS[model_type]['description']}")
    
    # Validate model availability
    if not validate_model_availability(model_type):
        print(f"\nError: Model '{model_type}' is not ready for use")
        print("   Please check the requirements above or choose a different model")
        sys.exit(1)
    
    print(f"Using model: {model_type}")
    
    # Check input files
    sentiment_csv = args.sentiment_csv
    
    if not os.path.exists(sentiment_csv):
        print(f"Error: Sentiment CSV not found: {sentiment_csv}")
        sys.exit(1)
    
    print(f"Input files validated")
    
    # Initialize pipeline
    try:
        print(f"Initializing {model_type} pipeline...")
        pipeline = SummarizationPipeline(model_type=model_type)
        print("Pipeline initialized successfully")
    except Exception as e:
        print(f"Error: Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run pipeline
    try:
        print(f"Running pipeline with {model_type}...")
        results = pipeline.run_pipeline(sentiment_csv)
        
        print("\n=== RESULTS ===")
        print(f"Generated {len(results['category_articles'])} category guides")

        # Save results
        output_dir = f"deploy/category-pages/outputs/{model_type}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")

        for category, article_data in results['category_articles'].items():
            filename = f"{output_dir}/{category.replace(' ', '_').lower()}_data.json"
            cleaned_data = clean_data_recursively(article_data)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, default=json_serializer)
            print(f"   {category}")

        # Save pipeline stats
        with open(f"{output_dir}/pipeline_stats.json", 'w', encoding='utf-8') as f:
            json.dump(results['stats'], f, indent=2)

        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"View results at: deploy/category-pages/index.html")

    except FileNotFoundError as e:
        print(f"Error: Missing file: {e}")
        print(f"\nExpected files:")
        print(f"   1. {sentiment_csv} (sentiment analysis with review text)")
        print(f"\nMake sure your teammate has provided the CSV file.")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
