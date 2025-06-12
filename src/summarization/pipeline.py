"""
Simple CSV-Based Summarization Pipeline
Loads CSV files from teammate's work and generates summaries/articles
"""

import time
import gc
from typing import List, Dict
import pandas as pd
import torch
from .models import generate_comparison_article, create_model_pipeline


class SummarizationPipeline:
    """Simple CSV-based pipeline for generating category summaries and articles"""

    def __init__(self, model_type: str = "gemma"):
        self.model_type = model_type
        print(f"Loading {model_type} model (this may take a moment)...")
        self.model_pipeline, self.model_config = create_model_pipeline(model_type)
        print(f"✓ Model {model_type} loaded and ready for inference")

    def cleanup(self):
        """Clean up model resources to free memory"""
        try:
            # Clear model references
            if hasattr(self, 'model_pipeline'):
                del self.model_pipeline
            if hasattr(self, 'model_config'):
                del self.model_config

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            print("✓ Model resources cleaned up")
        except (ImportError, AttributeError, RuntimeError) as e:
            print(f"Warning: Error during cleanup: {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

    def load_data(self, sentiment_csv: str, category_csv: str) -> pd.DataFrame:
        """Load and merge all data from CSV files"""
        print("Loading data from CSV files...")

        # Load sentiment analysis results (contains review text + sentiment)
        sentiment_df = pd.read_csv(sentiment_csv)
        print(f"Loaded {len(sentiment_df)} sentiment predictions "
              f"with review text")

        # Load category mapping
        category_df = pd.read_csv(category_csv)
        print(f"Loaded {len(category_df)} category mappings")

        # Handle different column names for product ID
        if 'product_id' in sentiment_df.columns:
            merge_on = 'product_id'
        elif 'id' in sentiment_df.columns:
            # Rename id to product_id for consistency
            sentiment_df = sentiment_df.rename(columns={'id': 'product_id'})
            merge_on = 'product_id'
        else:
            raise ValueError(
                "No product_id or id column found in sentiment data")

        # Merge categories with sentiment data
        final_df = category_df.merge(sentiment_df, on=merge_on)

        # Handle different sentiment column names
        if 'predicted_sentiment_SVC' in final_df.columns and 'sentiment' not in final_df.columns:
            final_df = final_df.rename(columns={'predicted_sentiment_SVC': 'sentiment'})
            print("Renamed predicted_sentiment_SVC to sentiment")

        # Handle different column names for category (cluster vs meta_category)
        if 'cluster' in final_df.columns and 'meta_category' not in final_df.columns:
            final_df['meta_category'] = final_df['cluster'].map({
                0: 'Fire TV & Streaming Devices',
                1: 'Charging & Accessories',
                2: 'Kindle Cases & Covers',
                3: 'Fire Tablets & Echo Speakers',
                4: 'E-Readers & Kindle Devices'
            })
            print("Mapped cluster numbers to category names")

        print(
            f"Final merged dataset: {len(final_df)} reviews with sentiment and categories")
        return final_df

    def run_pipeline(self, sentiment_csv: str,
                     category_csv: str) -> Dict[str, Dict]:
        """Run the complete pipeline with AI content generation"""

        # Load and merge data
        df = self.load_data(sentiment_csv, category_csv)

        # Generate category articles with AI content
        category_articles = self.generate_category_articles(df)

        # Generate summary statistics
        stats = self.generate_stats(df)

        return {
            'category_articles': category_articles,
            'stats': stats
        }

    def generate_category_articles(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate category articles with AI-generated content"""
        articles = {}

        # Group by meta-category
        categories = df['meta_category'].dropna().unique()
        print(f"Found {len(categories)} categories: {list(categories)}")

        for i, category in enumerate(categories):
            print(f"\n[{i+1}/{len(categories)}] Processing category: {category}")
            category_data = df[df['meta_category'] == category]

            # Filter out rows without sentiment data (NaN from left join)
            text_col = ('review_text' if 'review_text' in category_data.columns
                       else 'reviews.text')
            category_data_with_reviews = category_data.dropna(subset=[text_col])

            print(f"  - Found {len(category_data)} total products, "
                  f"{len(category_data_with_reviews)} with reviews for {category}")

            if len(category_data_with_reviews) < 3:  # Skip categories with too few reviews
                print(
                    f"  - Skipping {category} (too few reviews: {len(category_data)})")
                continue

            print(f"  - Getting top products for {category}...")
            # Get top 3 products by average rating and review count (use data with reviews)
            top_products = self.get_top_products_for_category(category_data_with_reviews)
            print(f"  - Found {len(top_products)} top products")

            if len(top_products) >= 1:
                print(f"  - Getting sample reviews for {category}...")
                # Get sample reviews for quote extraction
                sample_reviews = self.get_sample_reviews_for_products(
                    category_data_with_reviews, top_products)

                print(
                    f"  - Generating AI content for {category} (this may take a while)...")
                # Generate AI content with sample reviews
                ai_start = time.time()
                ai_summary = generate_comparison_article(
                    self.model_pipeline, self.model_config, top_products, category, sample_reviews)
                ai_time = time.time() - ai_start

                # Build structured data ourselves
                avg_rating = category_data_with_reviews['rating'].mean()
                sentiment_dist = category_data_with_reviews['sentiment'].value_counts().to_dict()

                # Extract pros and cons from sentiment analysis
                pros = self.extract_pros_from_reviews(sample_reviews, sentiment_dist)
                cons = self.extract_cons_from_reviews(sample_reviews)

                # Create structured article data
                article_data = {
                    'category': category,
                    'total_reviews': len(category_data_with_reviews),
                    'top_products': top_products,
                    'sample_reviews': sample_reviews,
                    'stats': {
                        'avg_rating': avg_rating,
                        'sentiment_distribution': sentiment_dist
                    },
                    'buying_guide': {
                        'title': f"{category} Buying Guide - Expert Analysis",
                        'pros': pros,
                        'cons': cons,
                        'recommendation': ai_summary,  # Just the LLM-generated recommendation text
                        'rating': round(avg_rating, 1),
                        'total_reviews': len(category_data_with_reviews),
                        'generation_time': ai_time,
                        'model_used': self.model_type
                    }}

                articles[category] = article_data
                print(f"  - Generated AI content for {category} category "
                      f"({len(ai_summary)} chars in {ai_time:.2f}s)")
            else:
                print(
                    f"  - Skipping {category} (insufficient top products: {len(top_products)})")

        return articles

    def extract_pros_from_reviews(self, sample_reviews: Dict, sentiment_dist: Dict) -> list:
        """Extract pros from positive customer reviews"""
        pros = []
        positive_pct = round((sentiment_dist.get('positive', 0) /
                             sum(sentiment_dist.values())) * 100)

        # Find common positive themes from sample reviews
        positive_keywords = []
        for _, reviews in sample_reviews.items():
            for review in reviews:
                if review['sentiment'] == 'positive':
                    text = review['text'].lower()
                    # Look for positive indicators
                    if 'love' in text or 'great' in text:
                        if 'children' in text or 'kids' in text:
                            positive_keywords.append('family-friendly')
                        if 'travel' in text or 'portable' in text:
                            positive_keywords.append('portable')
                        if 'easy' in text:
                            positive_keywords.append('user-friendly')
                        if 'battery' in text:
                            positive_keywords.append('good-battery')

        # Generate pros based on data
        if positive_pct >= 75:
            pros.append(f"High customer satisfaction ({positive_pct}% positive reviews)")

        if 'family-friendly' in positive_keywords:
            pros.append("Customers frequently mention it's great for children and family use")

        if 'portable' in positive_keywords:
            pros.append("Praised for portability and travel convenience")

        if 'user-friendly' in positive_keywords:
            pros.append("Consistently described as easy to use")

        if not pros:  # Fallback
            pros.append(f"Generally positive customer feedback ({positive_pct}% positive)")

        return pros[:3]  # Limit to 3 pros

    def extract_cons_from_reviews(self, sample_reviews: Dict) -> list:
        """Extract cons from negative/neutral customer reviews"""
        cons = []

        # Find common negative themes
        negative_keywords = []
        for _, reviews in sample_reviews.items():
            for review in reviews:
                if review['sentiment'] in ['negative', 'neutral']:
                    text = review['text'].lower()
                    if 'ads' in text or 'advertisement' in text:
                        negative_keywords.append('ads')
                    if 'elderly' in text or 'senior' in text:
                        negative_keywords.append('senior-issues')
                    if 'slow' in text or 'laggy' in text:
                        negative_keywords.append('performance')
                    if 'cheap' in text or 'flimsy' in text:
                        negative_keywords.append('build-quality')

        # Generate cons based on complaints
        if 'ads' in negative_keywords:
            cons.append("Some users find advertisements disruptive")

        if 'senior-issues' in negative_keywords:
            cons.append("May not be ideal for elderly or less tech-savvy users")

        if 'performance' in negative_keywords:
            cons.append("Performance issues mentioned in some reviews")

        if 'build-quality' in negative_keywords:
            cons.append("Build quality concerns from some customers")

        if not cons:  # Fallback
            cons.append("Limited negative feedback available")

        return cons[:2]  # Limit to 2 cons

    def get_sample_reviews_for_products(self,
                                        category_data: pd.DataFrame,
                                        top_products: List[Dict]) -> Dict[str,
                                                                          List[Dict]]:
        """Get sample reviews for each top product to provide quotes for LLM"""
        sample_reviews = {}

        for product in top_products:
            product_reviews = category_data[category_data['product_id']
                                            == product['product_id']]

            # Get a mix of positive, negative, and neutral reviews
            positive_reviews = product_reviews[product_reviews['sentiment'] == 'positive'].head(
                3)
            negative_reviews = product_reviews[product_reviews['sentiment'] == 'negative'].head(
                2)
            neutral_reviews = product_reviews[product_reviews['sentiment'] == 'neutral'].head(
                1)

            # Combine and convert to list of dicts
            all_reviews = pd.concat(
                [positive_reviews, negative_reviews, neutral_reviews])

            # Handle different column name formats
            text_col = 'review_text' if 'review_text' in all_reviews.columns else 'reviews.text'
            rating_col = 'rating' if 'rating' in all_reviews.columns else 'reviews.rating'

            sample_reviews[product['name']] = all_reviews[
                [text_col, rating_col, 'sentiment']
            ].rename(columns={text_col: 'text', rating_col: 'rating'}).to_dict('records')

        return sample_reviews

    def get_top_products_for_category(
            self, category_data: pd.DataFrame) -> List[Dict]:
        """Get top 3 products for a category based on ratings and sentiment"""

        # Group by product and calculate metrics
        # Handle different column name formats
        rating_col = 'rating' if 'rating' in category_data.columns else 'reviews.rating'
        text_col = 'review_text' if 'review_text' in category_data.columns else 'reviews.text'

        # Get product names for each product_id
        product_info = category_data.groupby('product_id').agg({
            rating_col: ['mean', 'count'],
            'sentiment': lambda x: (x == 'positive').sum() / len(x),
            text_col: 'first'  # Get first review text to extract product name if needed
        }).round(2)

        # Flatten column names
        product_info.columns = [
            'avg_rating',
            'review_count',
            'positive_ratio',
            'sample_text']

        # Filter products with at least 1 review (lowered from 2)
        product_info = product_info[product_info['review_count'] >= 1]

        # Calculate combined score (rating weighted more than sentiment)
        product_info['combined_score'] = (
            product_info['avg_rating'] * 0.7 +
            product_info['positive_ratio'] * 5 * 0.3
        )

        # Sort by combined score and take top 3
        top_products_data = product_info.sort_values(
            'combined_score', ascending=False).head(3)

        # Convert to list of dicts for LLM processing
        result = []
        for product_id, metrics in top_products_data.iterrows():
            # Get actual product name from the category data (now includes
            # 'name' column)
            product_name_row = category_data[category_data['product_id'] == product_id]
            if 'name' in category_data.columns and len(product_name_row) > 0:
                product_name = product_name_row['name'].iloc[0]
            else:
                # Fallback to product_id if name not available
                product_name = product_id

            result.append({
                'product_id': product_id,
                'name': product_name,
                'avg_rating': metrics['avg_rating'],
                'review_count': int(metrics['review_count']),
                'positive_ratio': metrics['positive_ratio'],
                'combined_score': metrics['combined_score']
            })

        return result

    def generate_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        return {
            'total_reviews': len(df),
            'unique_products': df['name'].nunique(),
            'categories_covered': df['meta_category'].nunique(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'average_rating': df['rating'].mean()
        }
