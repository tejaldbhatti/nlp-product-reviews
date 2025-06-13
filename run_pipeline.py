"""
Simple CSV-Based Summarization Pipeline
Run this after the sentiment and category CSV files are ready.
"""
import os
import math
import json
from src.summarization.pipeline import SummarizationPipeline


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

def main():
    """Main function to run the NLP product review summarization pipeline."""
    sentiment_csv = "results/sentiment_results.csv"
    category_csv = "results/category_mapping.csv"

    print("=== ROBOREVIEWS SUMMARIZATION PIPELINE ===")

    # models available: gemma-2b, mistral, qwen, qwen-finetuned, gemini-pro-flash
    model_type = "gemma-2b"  # Change this to the desired model type
    print(f"Using model: {model_type}")
    pipeline = SummarizationPipeline(model_type=model_type)

    try:
        results = pipeline.run_pipeline(sentiment_csv, category_csv)
        print("\n=== RESULTS ===")
        print(f"Generated {len(results['category_articles'])} category guides")

        output_dir = f"deploy/category-pages/outputs/{model_type}"
        os.makedirs(output_dir, exist_ok=True)

        for category, article_data in results['category_articles'].items():
            filename = f"{output_dir}/{category.replace(' ', '_').lower()}_data.json"
            cleaned_data = clean_data_recursively(article_data)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, default=json_serializer)

        with open(f"{output_dir}/pipeline_stats.json", 'w', encoding='utf-8') as f:
            json.dump(results['stats'], f, indent=2)

        print("Pipeline completed successfully!")

    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        print("\nExpected files:")
        print(f"1. {sentiment_csv} (sentiment analysis with review text)")
        print(f"2. {category_csv} (category mapping)")
        print("\nMake sure your teammate has provided both CSV files.")

if __name__ == "__main__":
    main()
