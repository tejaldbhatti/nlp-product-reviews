Product Review Analysis & Recommendation System
🚀 Overview

This project builds a product review website powered by NLP models that aggregate customer feedback from multiple sources. It helps businesses:

    Analyze review sentiment

    Organize products into meaningful categories

    Recommend top products effectively using AI

🔍 Main Features

    Sentiment Classification: Categorize reviews as Positive, Neutral, or Negative.

    Product Category Clustering: Group products into 4–6 meaningful categories.

    Generative AI Summarization & Recommendations: Summarize reviews and recommend top products with Large Language Models (LLMs).

📦 Project Parts
Part 1: Sentiment Classification

Objective:
Classify Amazon product reviews into Positive, Neutral, or Negative sentiments to help businesses understand customer opinions.

Dataset:
Amazon reviews from 1429_1.csv, refined using VADER sentiment analysis.

Methodology:

    Extract TF-IDF features (top 5000) from review text.

    Train a Support Vector Machine (SVM) classifier with balanced class weights.

Results:

    Achieved 85.68% accuracy.

    Output file contains reviews with predicted sentiments and confidence scores.

Output:
product_reviews_sentiment_and_confidence.csv — reviews plus predicted sentiment and confidence.

How to Run:

    Upload 1429_1.csv.

    Run the notebook sentiment_analysis_tfidf_svm.ipynb.

Part 2: Product Category Clustering

Objective:
Group products into 4–6 meaningful categories for simplified organization and targeted analysis.

Dataset:
Product names and hierarchical categories extracted from 1429_1.csv.

Methodology:

    Clean data (handle missing values and duplicates).

    Create combined text from product name and category for clustering.

    Generate embeddings using pretrained all-MiniLM-L6-v2 model.

    Apply clustering algorithms: KMeans, Agglomerative, HDBSCAN.

    Use UMAP for dimensionality reduction and cluster visualization.

    Evaluate clusters with Silhouette scores (~0.11–0.13) and qualitative analysis.

Results:

    Identified meaningful clusters such as Power Adapters, Kindle E-readers, Fire TV devices.

Output:

    aggregated_reviews_cluster.csv — aggregated review data with cluster assignments.

    category_cluster_with_id.csv — product categories with cluster IDs.

How to Run:

    Place 1429_1.csv next to clustering_model_1.ipynb.

    Run all cells in the notebook.

Part 3: Generative AI Summarization & Recommendations (Upcoming)

Objective:
Summarize reviews and recommend top products per category using Large Language Models (LLMs).

Dataset:
Aggregated reviews and cluster data from previous parts.

Methodology:

    Utilize state-of-the-art LLMs for generating summaries and recommendations.

    Avoid classical summarization techniques.

Results:

    (To be developed) Aim to deliver concise and insightful product summaries and recommendations.

Output:
(To be defined in future releases)

How to Run:
(To be provided once implemented)
