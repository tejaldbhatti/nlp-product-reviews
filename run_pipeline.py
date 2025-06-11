"""
Simple CSV-Based Summarization Pipeline
Run this after the sentiment and category CSV files are ready.
"""

import json
from src.summarization.pipeline import SummarizationPipeline


def main():
    """Main function to run the NLP product review summarization pipeline."""
    sentiment_csv = "results/sentiment_results.csv"
    category_csv = "results/category_mapping.csv"

    print("=== ROBOREVIEWS SUMMARIZATION PIPELINE ===")
    print()

    # Model options (fastest to slowest, available without gating):
    # gemma-2b        (2B params) - Good balance of speed and quality
    # phi3            (3.8B params) - Proven performance
    # mistral         (7B params) - High quality
    # gemma           (Gemma 3 4B params) - Current default
    # mistral-finetuned (7B params) - Fine-tuned for summarization tasks

    model_type = "mistral-finetuned"
    print(f"Using model: {model_type}")

    # Initialize pipeline
    pipeline = SummarizationPipeline(model_type=model_type)

    # Run the pipeline
    try:
        results = pipeline.run_pipeline(sentiment_csv, category_csv)

        print("\n=== PIPELINE RESULTS ===")
        print(f"Statistics: {results['stats']}")
        print(f"Generated articles for {len(results['category_articles'])} categories")

        # Save results to deployment directory
        output_dir = f"deploy/category-pages/outputs/{model_type}"
        print(f"\nSaving results to {output_dir}/...")

        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save category data as JSON files
        for category, article_data in results['category_articles'].items():
            filename = f"{output_dir}/{category.replace(' ', '_').lower()}_data.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, default=str)
            print(f"Saved: {filename}")

        # Save stats
        with open(f"{output_dir}/pipeline_stats.json", 'w', encoding='utf-8') as f:
            json.dump(results['stats'], f, indent=2)

        print("\nPipeline completed successfully!")

    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        print("\nExpected files:")
        print(f"1. {sentiment_csv} (sentiment analysis with review text)")
        print(f"2. {category_csv} (category mapping)")
        print("\nMake sure your teammate has provided both CSV files.")

if __name__ == "__main__":
    main()
