# Product Review Analysis & Recommendation System

## Overview

This project analyzes Amazon product reviews to generate comprehensive product recommendation guides using natural language processing and machine learning. The system combines sentiment analysis, product clustering, and generative AI to create professional buying guides for different product categories.

## Features

- **Sentiment Analysis**: Classifies reviews as positive, negative, or neutral using TF-IDF features and SVM
- **Product Clustering**: Automatically groups products into meaningful categories using sentence embeddings
- **AI-Generated Recommendations**: Creates comprehensive buying guides using multiple language models
- **Multi-Model Support**: Supports various models including fine-tuned specialized models
- **Web Interface**: Generates HTML pages for easy viewing of recommendations

## Project Components

### 1. Sentiment Classification

**Objective**: Classify Amazon product reviews to understand customer sentiment patterns.

**Methodology**:
- TF-IDF feature extraction (top 5000 features)
- Support Vector Machine classifier with balanced class weights
- VADER sentiment analysis for initial labeling

**Results**: Achieved 85.68% accuracy on the test set.

**Output**: `results/sentiment_results.csv` containing reviews with predicted sentiment and confidence scores.

### 2. Product Category Clustering

**Objective**: Group products into 4-6 meaningful categories for organized analysis.

**Methodology**:
- Data cleaning and preprocessing
- Sentence embeddings using all-MiniLM-L6-v2 model
- Multiple clustering algorithms (KMeans, Agglomerative, HDBSCAN)
- UMAP dimensionality reduction for visualization
- Cluster evaluation using silhouette scores and manual inspection

**Results**: Identified distinct product categories including E-readers, Fire TV devices, Kindle accessories, tablets, and charging accessories.

**Output**: `results/aggregated_reviews_cluster.csv` containing products with cluster assignments.

### 3. AI-Generated Product Recommendations

**Objective**: Generate comprehensive, professional buying guides for each product category.

**Methodology Evolution**:
1. **Phase 1**: Few-shot prompting with carefully crafted prompts
2. **Phase 2**: Multi-model comparison and optimization
3. **Phase 3**: Fine-tuning specialized models for consistent, high-quality output

**Available Models**:
- `gemma-2b`: Fast, general-purpose model (2B parameters)
- `gemma-3`: Efficient instruction-tuned model (1B parameters, MPS optimized)
- `mistral`: Balanced performance model (7B parameters)
- `qwen`: High-quality base model (7B parameters)
- `qwen-finetuned`: Specialized for product recommendations (7B parameters)
- `gemma-3-finetuned`: Specialized, efficient model (1B parameters, MPS optimized)
- `gemini-pro-flash`: Cloud API model (very fast, high quality)

**Output**: Professional buying guides with structured analysis, pros/cons, and recommendations for each product category.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-product-reviews
```

2. Create a virtual environment:
```bash
python -m venv venv_nlp
source venv_nlp/bin/activate  # On Windows: venv_nlp\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
GOOGLE_API_KEY=your_google_api_key_here  # Only needed for gemini-pro-flash
```

## Usage

### Running the Pipeline

The main pipeline generates product recommendation guides using pre-trained models:

```bash
# List available models
python run_pipeline.py --list-models

# Run with a specific model
python run_pipeline.py gemma-3-finetuned

# Run with custom data file
python run_pipeline.py qwen --sentiment-csv path/to/your/sentiment_data.csv
```

**Available Models**:
- `gemma-3-finetuned`: Recommended for best balance of speed and quality
- `qwen-finetuned`: High-quality specialized model
- `gemini-pro-flash`: Fastest option (requires Google API key)
- `gemma-2b`, `gemma-3`, `mistral`, `qwen`: Base models for comparison

**Output**: Results are saved to `deploy/category-pages/outputs/{model_name}/` with:
- Individual category JSON files with detailed recommendations
- Pipeline statistics and metadata
- View results at `deploy/category-pages/index.html`

### Live Demo

When code is pushed to the GitHub repository, a GitHub Actions workflow automatically deploys the generated results to GitHub Pages. You can view the live demo at: https://malibio.github.io/nlp-product-reviews/

The deployment is configured via GitHub Actions YAML files and updates automatically with each push to the main branch.

### Training Fine-Tuned Models

The project includes fine-tuned models optimized for product recommendations. The training process follows these steps:

#### 1. Prepare Training Data

Training data should be in JSON format with instruction-response pairs:
```json
[
  {
    "instruction": "Create a product recommendation guide for Fire TV & Streaming Devices based on customer reviews and ratings.",
    "response": "### **The Fire TV Experience: What Users Are Saying...**\n\nDetailed analysis and recommendations...",
    "category": "Fire TV & Streaming Devices"
  }
]
```

The project includes 173 high-quality training examples in `data/synthetic/fine_tuning_prompt_response.json`.

#### 2. Training Configuration

Fine-tuning uses LoRA (Low-Rank Adaptation) for efficient training:
- **Model**: google/gemma-3-1b-it (instruction-tuned base)
- **Method**: LoRA fine-tuning (trainable parameters: ~0.65% of total)
- **Training**: 3 epochs, batch size 1, learning rate 2e-4
- **Hardware**: Optimized for Apple Silicon MPS

#### 3. Training Process

The training approach evolved through multiple iterations:
1. **Initial approach**: Complex training module with label masking
2. **Final approach**: Simplified training following successful patterns from other models

Key technical decisions:
- Use instruction-tuned base model (google/gemma-3-1b-it)
- Train on entire sequence to learn conversation flow
- Conservative generation settings for MPS stability
- Higher LoRA rank (r=16) and alpha (32) for better adaptation

#### 4. Model Integration

Fine-tuned models are automatically integrated into the pipeline:
- Models saved as PEFT adapters (~26MB vs 2GB full model)
- Seamless switching between base and fine-tuned models
- Specialized generation parameters for optimal output

### Understanding the Results

Generated buying guides include:
- **Market Overview**: Introduction to the product category
- **Strengths Analysis**: What customers consistently praise
- **Concerns Analysis**: Common issues and red flags
- **Expert Recommendations**: Actionable buying advice
- **Product Rankings**: Top products with ratings and review counts

## Project Structure

```
├── data/
│   ├── processed/           # Processed data files
│   └── synthetic/          # Training data for fine-tuning
├── deploy/
│   └── category-pages/     # Generated web interface
├── notebooks/              # Jupyter notebooks for analysis
├── results/               # Output files from processing
├── roboreviews-gemma-simple/    # Fine-tuned Gemma model
├── roboreviews-qwen-finetuned/  # Fine-tuned Qwen model
├── src/
│   ├── clustering_model.py      # Product clustering implementation
│   ├── sentiment_analysis_model.py  # Sentiment classification model
│   └── summarization/          # Core pipeline code
├── run_pipeline.py        # Main entry point
└── requirements.txt       # Dependencies
```

## Key Files

- `run_pipeline.py`: Main script for running the recommendation pipeline
- `src/summarization/pipeline.py`: Core pipeline logic
- `src/summarization/models.py`: Model configurations and interfaces
- `src/clustering_model.py`: Product clustering implementation
- `src/sentiment_analysis_model.py`: Sentiment classification model
- `results/sentiment_results.csv`: Sentiment analysis results
- `results/aggregated_reviews_cluster.csv`: Clustered product data
- `deploy/category-pages/index.html`: Web interface for viewing results

## Technical Details

### Model Architecture

The system uses a factory pattern for model management, supporting:
- Local models (Gemma, Mistral, Qwen) with HuggingFace integration
- API-based models (Gemini) for cloud inference
- Fine-tuned adapters using PEFT for specialization

### Apple Silicon Optimization

Special optimizations for Apple Silicon (M1/M2/M3) chips:
- MPS (Metal Performance Shaders) acceleration
- Conservative inference settings for numerical stability
- Memory-efficient model loading and caching

### Data Processing Pipeline

1. **Sentiment Analysis**: TF-IDF + SVM classification
2. **Clustering**: Sentence embeddings + KMeans clustering
3. **Content Generation**: Template-based prompting or fine-tuned generation
4. **Post-processing**: Cleaning and formatting of generated content

## Development Notes

### Methodology Evolution

The project evolved through three main phases:
1. **Few-shot prompting**: Initial implementation with complex prompts
2. **Multi-model comparison**: Systematic evaluation of different models
3. **Fine-tuning specialization**: Custom models trained for the specific task

### Fine-Tuning Insights

Key learnings from the fine-tuning process:
- Instruction-tuned base models are essential for dialogue tasks
- Simple training approaches often outperform complex ones
- Proper tokenizer handling is critical for generation quality
- Conservative generation settings improve stability on Apple Silicon

### Quality Improvements

The fine-tuned models provide:
- Consistent output structure and formatting
- Professional writing style and tone
- Reduced prompt engineering requirements
- Specialized knowledge for product recommendations

## Team

This project was developed by:
- **Tehal Bhatti**
- **Michael Libio**

## Contributing

1. Follow the existing code structure and naming conventions
2. Test changes with multiple models to ensure compatibility
3. Update documentation for any new features or models
4. Ensure Apple Silicon compatibility for local development

## License

This project is for educational and research purposes.