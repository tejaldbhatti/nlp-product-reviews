"""
Simple CSV-Based Summarization Pipeline
Loads CSV files from teammate's work and generates summaries/articles
"""

import time
import gc
from typing import List, Dict
import pandas as pd
import torch
from .models import create_model_pipeline, clean_recommendation_text, generate_text


class SummarizationPipeline:
    """Simple CSV-based pipeline for generating category summaries and articles"""

    def __init__(self, model_type: str = "gemma"):
        self.model_type = model_type
        self.model_pipeline, self.model_config = create_model_pipeline(model_type)

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
        except (ImportError, AttributeError, RuntimeError):
            pass

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

    def load_cluster_data(self) -> pd.DataFrame:
        """Load aggregated cluster data"""
        cluster_df = pd.read_csv('results/aggregated_reviews_cluster.csv')
        cluster_df = cluster_df.loc[:, ~cluster_df.columns.duplicated()]
        cluster_df = self.clean_product_names(cluster_df)
        cluster_df = self.deduplicate_products(cluster_df)

        if 'kmeans_cluster' in cluster_df.columns and 'meta_category' not in cluster_df.columns:
            cluster_df['meta_category'] = cluster_df['kmeans_cluster'].map({
                0: 'Fire TV & Streaming Devices',
                1: 'Charging & Accessories',
                2: 'Kindle Cases & Covers',
                3: 'Fire Tablets & Echo Speakers',
                4: 'E-Readers & Kindle Devices'
            })

        return cluster_df

    def clean_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean corrupted product names"""
        if 'name' in df.columns:
            df = df.copy()
            def extract_clean_name(name):
                if pd.isna(name):
                    return name

                if ',,\r\n' in name or ',,,' in name:
                    clean_parts = name.split(',,,')[0].split('\r\n')[0].strip()
                    if clean_parts and len(clean_parts) > 5:
                        return clean_parts
                    parts = name.replace(',,,\r\n', '|||').replace(',,,', '|||').split('|||')
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 5:
                            return part

                return name

            df['name'] = df['name'].apply(extract_clean_name)
            df = df[df['name'].str.len() > 5]
            df['name'] = df['name'].str.replace(r'^"', '', regex=True)
            df['name'] = df['name'].str.replace(r'"$', '', regex=True)
            df['name'] = df['name'].str.strip()
            df['name'] = df['name'].str.replace(r',+$', '', regex=True)
            def clean_remaining_issues(name):
                if pd.isna(name):
                    return name
                if ',,' in name:
                    parts = [part.strip() for part in name.split(',') if part.strip()]
                    return parts[0] if parts else name
                return name

            df['name'] = df['name'].apply(clean_remaining_issues)

        return df

    def deduplicate_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve products appearing in multiple clusters"""
        return df.drop_duplicates(subset=['product_id'], keep='first')


    def extract_quotable_reviews(
            self, products: List[Dict], sentiment_df: pd.DataFrame) -> Dict:
        """Link products to individual review quotes via product_id"""
        quotes_by_product = {}

        for product in products:
            product_id = product['product_id']
            quotes_by_product[product_id] = self._extract_product_quotes(
                product_id, sentiment_df)

        return quotes_by_product

    def _extract_product_quotes(self, product_id: str, sentiment_df: pd.DataFrame) -> Dict:
        """Extract quotes for a single product"""
        product_reviews = sentiment_df[sentiment_df['product_id'] == product_id]

        if len(product_reviews) == 0:
            return {'positive_quotes': [], 'warning_quotes': []}

        positive_reviews = product_reviews[
            (product_reviews['predicted_sentiment_SVC'] == 'positive') &
            (product_reviews['rating'] >= 4.0)
        ]

        negative_reviews = product_reviews[
            (product_reviews['predicted_sentiment_SVC'] == 'negative') &
            ((product_reviews['rating'] <= 3.0) |
             (product_reviews['rating'].isna()))
        ]

        positive_quotes = self._extract_positive_quotes(positive_reviews)
        warning_quotes = self._extract_warning_quotes(negative_reviews)

        return {
            'positive_quotes': positive_quotes,
            'warning_quotes': warning_quotes
        }

    def _filter_by_length(self, reviews_df):
        """Filter reviews by text length"""
        text_col = 'reviews.text' if 'reviews.text' in reviews_df.columns else 'review_text'
        if text_col in reviews_df.columns:
            return reviews_df[
                (reviews_df[text_col].str.len() >= 20) &
                (reviews_df[text_col].str.len() <= 500)
            ]
        return reviews_df

    def _extract_positive_quotes(self, positive_reviews) -> List[Dict]:
        """Extract positive quotes from reviews"""
        positive_filtered = self._filter_by_length(positive_reviews)
        positive_quotes = []

        if len(positive_filtered) > 0:
            top_positive = positive_filtered.nlargest(2, 'rating')
            for _, review in top_positive.iterrows():
                text_col = 'reviews.text' if 'reviews.text' in review else 'review_text'
                positive_quotes.append({
                    'text': review[text_col],
                    'rating': review['rating']
                })

        return positive_quotes

    def _extract_warning_quotes(self, negative_reviews) -> List[Dict]:
        """Extract warning quotes from reviews"""
        negative_filtered = self._filter_by_length(negative_reviews)
        warning_quotes = []

        if len(negative_filtered) > 0:
            bottom_negative = negative_filtered.nsmallest(2, 'rating')
            for _, review in bottom_negative.iterrows():
                text_col = 'reviews.text' if 'reviews.text' in review else 'review_text'
                warning_quotes.append({
                    'text': review[text_col],
                    'rating': review['rating']
                })

        return warning_quotes

    def run_pipeline(self, sentiment_csv: str) -> Dict[str, Dict]:
        """Run the complete pipeline with AI content generation"""

        # Load cluster data using new approach
        df = self.load_cluster_data()

        sentiment_df = pd.read_csv(sentiment_csv)
        category_articles = self.generate_category_articles(df, sentiment_df)

        # Generate summary statistics
        stats = self.generate_stats(df)

        return {
            'category_articles': category_articles,
            'stats': stats
        }

    def generate_category_articles(
            self, df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> Dict[str, str]:
        """Generate category articles with AI-generated content"""
        articles = {}
        categories = df['meta_category'].dropna().unique()

        for category in categories:
            category_data = df[df['meta_category'] == category]

            if len(category_data) < 1:
                continue

            enhanced_products = self.get_enhanced_products_for_category(category_data)

            if len(enhanced_products['top_picks']) >= 1:
                all_products = (enhanced_products['top_picks'] +
                               enhanced_products['avoid_products'])
                quotes = self.extract_quotable_reviews(all_products, sentiment_df)
                sample_reviews = self.get_sample_reviews_from_sentiment_data(
                    sentiment_df, all_products)

                ai_start = time.time()
                customer_insights = self.extract_customer_insights_for_prompts(
                    sample_reviews, enhanced_products)
                customer_insights['top_product_names'] = [
                    p['name'] for p in enhanced_products['top_picks'][:2]]
                customer_insights['avoid_product_names'] = [
                    p['name'] for p in enhanced_products.get('avoid_products', [])[:1]]

                model_limits = self.get_model_prompt_limits()

                if model_limits['use_chunked']:
                    review_sections = self.generate_comprehensive_review(
                        category, customer_insights)
                    comprehensive_content = self.assemble_reviewer_content(
                        review_sections, category)
                    comprehensive_content = clean_recommendation_text(comprehensive_content)
                else:
                    comprehensive_content = self.generate_unified_review(
                        category, customer_insights)
                    comprehensive_content = clean_recommendation_text(comprehensive_content)

                focused_content = {'category_summary': comprehensive_content}
                ai_time = time.time() - ai_start
                ai_summary = focused_content.get(
                    'category_summary', f'{category} products analyzed from customer reviews.')

                avg_rating = category_data['avg_rating'].mean()
                total_reviews = (
                    category_data['positive_review_count'].sum() +
                    category_data['negative_review_count'].sum() +
                    category_data['neutral_review_count'].sum())

                sentiment_dist = {
                    'positive': category_data['positive_review_count'].sum(),
                    'negative': category_data['negative_review_count'].sum(),
                    'neutral': category_data['neutral_review_count'].sum()}

                pros, cons = self.extract_insights_from_reviewer_content(
                    ai_summary, sentiment_dist)

                # Create enhanced structured article data
                article_data = {
                    'category': category,
                    'metadata': {
                        'total_products': len(category_data),
                        'total_reviews': int(total_reviews),
                        'avg_rating': round(avg_rating, 2),
                        'generation_info': {
                            'generation_time': ai_time,
                            'model_used': self.model_type,
                            'content_type': 'enhanced_reviewer_style'
                        }
                    },
                    'recommendations': {
                        'top_picks': [],
                        'avoid_products': []
                    },
                    'customer_insights': {
                        'why_customers_choose': ai_summary,
                        'main_concerns': 'Analyzed from customer feedback patterns',
                        'positive_themes': pros,
                        'negative_themes': cons
                    },
                    # Legacy fields for compatibility
                    'top_products': enhanced_products['top_picks'],
                    'sample_reviews': sample_reviews,
                    'stats': {
                        'avg_rating': avg_rating,
                        'sentiment_distribution': sentiment_dist
                    },
                    'buying_guide': {
                        'title': f"{category} Buying Guide - Expert Review Analysis",
                        'pros': pros,
                        'cons': cons,
                        'recommendation': ai_summary,  # Full reviewer content
                        'rating': round(avg_rating, 1),
                        'total_reviews': int(total_reviews),
                        'generation_time': ai_time,
                        'model_used': self.model_type,
                        'content_style': 'comprehensive_reviewer'
                    }
                }

                # Add enhanced product data with quotes (no extra LLM-generated content)
                for product in enhanced_products['top_picks']:
                    product_quotes = quotes.get(
                        product['product_id'],
                        {'positive_quotes': [], 'warning_quotes': []})

                    article_data['recommendations']['top_picks'].append({
                        'product_id': product['product_id'],
                        'name': product['name'],
                        'rating': (
                            float(product['avg_rating'])
                            if not pd.isna(product['avg_rating']) else 3.0),
                        'review_count': product['review_count'],
                        'positive_quotes': product_quotes['positive_quotes']
                    })

                for product in enhanced_products['avoid_products']:
                    product_quotes = quotes.get(
                        product['product_id'],
                        {'positive_quotes': [], 'warning_quotes': []})
                    article_data['recommendations']['avoid_products'].append({
                        'product_id': product['product_id'],
                        'name': product['name'],
                        'rating': (
                            float(product['avg_rating'])
                            if not pd.isna(product['avg_rating']) else 3.0),
                        'review_count': product['review_count'],
                        'why_avoid': (
                            "Lower customer satisfaction: "
                            f"{float(product['avg_rating']) if not pd.isna(product['avg_rating']) else 3.0:.1f}"
                            "/5 rating"),
                        'warning_quotes': product_quotes['warning_quotes']
                    })

                articles[category] = article_data

        return articles

    def extract_insights_from_reviewer_content(
            self, reviewer_content: str, sentiment_dist: Dict) -> tuple:
        """Extract pros and cons from comprehensive reviewer content"""
        pros = []
        cons = []

        # Extract from reviewer content sections
        content_lower = reviewer_content.lower()

        # Look for positive themes in "What's Working Really Well" section
        if 'working really well' in content_lower or 'working well' in content_lower:
            if 'easy' in content_lower or 'intuitive' in content_lower:
                pros.append("User-friendly design consistently praised by customers")
            if any(word in content_lower for word in ['family', 'kids', 'children']):
                pros.append("Excellent for family use and children")
            if 'sound' in content_lower or 'audio' in content_lower:
                pros.append("Strong audio performance for the price point")
            if 'value' in content_lower or 'price' in content_lower:
                pros.append("Good value proposition in the market")

        # Look for negative themes in "Red Flags" section
        if 'red flags' in content_lower or 'concerns' in content_lower:
            if 'durability' in content_lower or 'build' in content_lower:
                cons.append("Build quality and durability concerns reported")
            if 'ads' in content_lower or 'advertisement' in content_lower:
                cons.append("Advertisements can be disruptive for some users")
            if 'battery' in content_lower:
                cons.append("Battery life may not meet expectations")
            if 'slow' in content_lower or 'performance' in content_lower:
                cons.append("Performance issues mentioned in feedback")

        # Add sentiment-based fallbacks if we didn't extract enough
        total_sentiment = sum(sentiment_dist.values())
        positive_pct = round(
            (sentiment_dist.get('positive', 0) / total_sentiment) * 100
        ) if total_sentiment > 0 else 0

        if len(pros) == 0:
            if positive_pct >= 70:
                pros.append(f"High customer satisfaction ({positive_pct}% positive feedback)")
            pros.append("Generally positive customer experiences reported")

        if len(cons) == 0:
            cons.append("Some quality and performance concerns noted")

        return pros[:3], cons[:2]  # Limit results

    def get_sample_reviews_from_sentiment_data(
            self, sentiment_df: pd.DataFrame, top_products: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Get sample reviews from sentiment data for each top product"""
        sample_reviews = {}

        for product in top_products:
            product_id = product['product_id']
            product_reviews = sentiment_df[sentiment_df['product_id'] == product_id]

            if len(product_reviews) == 0:
                sample_reviews[product['name']] = []
                continue

            # Get a mix of positive, negative, and neutral reviews
            positive_reviews = product_reviews[
                product_reviews['predicted_sentiment_SVC'] == 'positive'].head(3)
            negative_reviews = product_reviews[
                product_reviews['predicted_sentiment_SVC'] == 'negative'].head(2)
            neutral_reviews = product_reviews[
                product_reviews['predicted_sentiment_SVC'] == 'neutral'].head(1)

            # Combine reviews
            all_reviews = pd.concat(
                [positive_reviews, negative_reviews, neutral_reviews])

            if len(all_reviews) == 0:
                sample_reviews[product['name']] = []
                continue

            # Convert to format expected by LLM
            reviews_list = []
            for _, review in all_reviews.iterrows():
                reviews_list.append({
                    'text': review['reviews.text'],
                    'rating': review['rating'],
                    'sentiment': review['predicted_sentiment_SVC'],
                    'product_id': review['product_id']
                })

            sample_reviews[product['name']] = reviews_list

        return sample_reviews

    def generate_single_prompt(self, prompt: str) -> str:
        """Generate text from a single focused prompt"""

        try:
            result = generate_text(
                self.model_pipeline,
                self.model_config["template"],
                self.model_config["extract_key"],
                prompt,
                self.model_config
            )
            return result.strip()
        except (RuntimeError, ValueError, OSError):
            return "Content generation failed"

    def get_reviewer_context(self, category: str) -> str:
        """Get category-specific reviewer expertise context"""
        contexts = {
            'Fire Tablets & Echo Speakers': 'smart home and family tech expert',
            'E-Readers & Kindle Devices': 'digital reading and e-ink specialist',
            'Charging & Accessories': 'mobile accessories and power solutions reviewer',
            'Kindle Cases & Covers': 'device protection and accessory expert',
            'Fire TV & Streaming Devices': 'streaming media and cord-cutting specialist'
        }
        return contexts.get(category, 'consumer electronics expert')

    def get_model_prompt_limits(self) -> Dict:
        """Get token limits based on model type for adaptive generation"""
        model_limits = {
            'gemma-2b': {
                'opening': 150,
                'strengths': 200,
                'concerns': 200,
                'recommendation': 150,
                'use_chunked': True
            },
            'gemma': {  # Alias for gemma-2b
                'opening': 150,
                'strengths': 200,
                'concerns': 200,
                'recommendation': 150,
                'use_chunked': True
            },
            'mistral': {
                'opening': 200,
                'strengths': 300,
                'concerns': 300,
                'recommendation': 200,
                'use_chunked': False
            },
            'qwen': {
                'opening': 200,
                'strengths': 300,
                'concerns': 300,
                'recommendation': 200,
                'use_chunked': False
            },
            'qwen-finetuned': {
                'opening': 180,
                'strengths': 250,
                'concerns': 250,
                'recommendation': 180,
                'use_chunked': True
            },
            'gemini-pro-flash': {
                'opening': 300,
                'strengths': 400,
                'concerns': 400,
                'recommendation': 300,
                'use_chunked': False
            }
        }

        # Return limits for current model, with fallback
        return model_limits.get(self.model_type.lower(), model_limits['mistral'])

    def generate_comprehensive_review(self, category: str, insights: Dict) -> Dict:
        """Generate review content in manageable chunks for small LLMs"""
        review_sections = {}

        # Section 1: Opening hook (small prompt)
        opening_prompt = f"""Write 2-3 sentences introducing {category} based on these \
customer feedback trends. Do not include phrases like "Okay" or "Here is". \
Start directly with the content.

POSITIVE: {insights['positive_examples']}
NEGATIVE: {insights['negative_examples']}

Write like a tech blogger setting the scene for readers."""
        review_sections['opening'] = self.generate_single_prompt(opening_prompt)

        # Section 2: Strengths analysis (focused prompt)
        strengths_prompt = f"""Based on positive customer feedback: \
{insights['positive_examples']}

Write one paragraph explaining what's working well in {category}. \
Mention specific benefits customers report. Be conversational and specific. \
Do not start with "Okay" or similar phrases."""
        review_sections['strengths'] = self.generate_single_prompt(strengths_prompt)

        # Section 3: Concerns analysis (focused prompt)
        concerns_prompt = f"""Based on negative customer feedback: \
{insights['negative_examples']}

Write one paragraph about the main problems customers face with {category}. \
Be specific about issues and write like you're warning readers. \
Do not start with "Okay" or similar phrases."""
        review_sections['concerns'] = self.generate_single_prompt(concerns_prompt)

        # Section 4: Final recommendation (synthesis prompt)
        recommendation_prompt = f"""For {category}, considering both the positives and \
negatives from customer reviews, write 2-3 sentences advising who should buy and \
who should avoid these products. Be direct and helpful. \
Do not start with "Okay" or similar phrases."""
        review_sections['recommendation'] = self.generate_single_prompt(recommendation_prompt)

        return review_sections

    def generate_unified_review(self, category: str, insights: Dict) -> str:
        """Generate comprehensive review content in single prompt for larger models"""

        unified_prompt = f"""Based on customer feedback analysis, write a comprehensive \
review in this exact structure. Do not include any introductory phrases like \
"Okay, here is..." or "Based on the data provided...". Start directly with the content.

**Customer Data:**
POSITIVE: {insights['positive_examples']}
NEGATIVE: {insights['negative_examples']}
TOP PRODUCTS: {insights.get('top_product_names', 'Various models')}
AVOID PRODUCTS: {insights.get('avoid_product_names', 'Lower-rated options')}

**Write the review following this structure:**

### **The {category} Landscape: A Comprehensive Look at Customer Feedback**
[2-3 sentences setting the scene based on the data]

#### **What's Working Really Well:**
[Paragraph highlighting 3-4 key strengths from positive feedback, mentioning specific products]

#### **The Red Flags to Watch:**
[Paragraph covering 2-3 main concerns from negative feedback, being specific about issues]

#### **My Recommendation:**
[Final paragraph with balanced advice on who should buy and who should avoid]

Write in a conversational, authoritative tone like a tech blogger. \
Use specific details from the customer data provided. \
Start directly with the header, no preamble."""

        return self.generate_single_prompt(unified_prompt)

    def assemble_reviewer_content(self, sections: Dict, category: str) -> str:
        """Assemble sections into cohesive reviewer-style content"""
        return f"""### **The {category} Landscape: A Comprehensive Look at Customer Feedback**

{sections['opening']}

#### **What's Working Really Well:**
{sections['strengths']}

#### **The Red Flags to Watch:**
{sections['concerns']}

#### **My Recommendation:**
{sections['recommendation']}"""

    def extract_customer_insights_for_prompts(
            self, sample_reviews: Dict, enhanced_products: Dict) -> Dict:
        """Extract enhanced customer insights from review data for LLM prompts"""
        positive_examples = []
        negative_examples = []

        # Enhanced positive insights extraction
        for _, reviews in sample_reviews.items():
            for review in reviews:
                if review['sentiment'] == 'positive' and len(positive_examples) < 4:
                    # Extract more meaningful quotes (longer, more descriptive)
                    text = review['text']
                    if len(text) > 50:  # Only substantial reviews
                        # Truncate but keep meaningful content
                        truncated = (text[:200] + '...' if len(text) > 200 else text)
                        positive_examples.append(
                            f"'{truncated}' ({review['rating']}/5)")

        # Enhanced negative insights extraction
        avoid_products = enhanced_products.get('avoid_products', [])
        for product in avoid_products:
            product_id = product.get('product_id')

            # Look for negative reviews of avoid products using product_id
            for _, reviews in sample_reviews.items():
                for review in reviews:
                    if (review.get('product_id') == product_id and
                        review['sentiment'] == 'negative' and
                        len(negative_examples) < 3):
                        text = review['text']
                        if len(text) > 30:  # Only substantial negative feedback
                            truncated = (text[:180] + '...'
                                       if len(text) > 180 else text)
                            negative_examples.append(
                                f"'{truncated}' ({review['rating']}/5)")

        # If we don't have enough negative examples from avoid products, get from all products
        if len(negative_examples) < 2:
            for _, reviews in sample_reviews.items():
                for review in reviews:
                    if (review['sentiment'] == 'negative' and
                        len(negative_examples) < 3 and
                        len(review['text']) > 30):
                        text = review['text']
                        truncated = (text[:180] + '...'
                                   if len(text) > 180 else text)
                        negative_examples.append(
                            f"'{truncated}' ({review['rating']}/5)")

        return {
            'positive_examples': (
                ' | '.join(positive_examples) if positive_examples
                else 'customers praise reliability, ease of use, and good value for money'),
            'negative_examples': (
                ' | '.join(negative_examples) if negative_examples
                else 'some concerns about durability and occasional performance issues')
        }

    def _calculate_avg_rating(self, product, review_counts: Dict) -> float:
        """Calculate average rating, handling NaN values"""
        avg_rating = product.get('avg_rating', 0)
        if pd.isna(avg_rating) or avg_rating == 0:
            total_reviews = review_counts['total']
            if total_reviews > 0:
                avg_rating = ((review_counts['positive'] * 5.0 +
                             review_counts['neutral'] * 3.0 +
                             review_counts['negative'] * 1.0) / total_reviews)
            else:
                avg_rating = 3.0
        return float(avg_rating) if not pd.isna(avg_rating) else 3.0

    def _handle_small_categories(self, category_data: pd.DataFrame) -> Dict:
        """Handle categories with fewer than 3 products"""
        products_list = []
        for _, product in category_data.iterrows():
            pos_count = product.get('positive_review_count', 0)
            neg_count = product.get('negative_review_count', 0)
            neu_count = product.get('neutral_review_count', 0)
            total_reviews = pos_count + neg_count + neu_count

            review_counts = {
                'positive': pos_count,
                'negative': neg_count,
                'neutral': neu_count,
                'total': total_reviews
            }
            avg_rating = self._calculate_avg_rating(product, review_counts)

            products_list.append({
                'product_id': product['product_id'],
                'name': product['name'],
                'avg_rating': avg_rating,
                'review_count': total_reviews,
                'positive_ratio': pos_count / total_reviews if total_reviews > 0 else 0,
                'composite_score': avg_rating,
                'pos_count': pos_count,
                'neg_count': neg_count
            })

        return {'top_picks': products_list, 'avoid_products': []}

    def get_enhanced_products_for_category(
            self, category_data: pd.DataFrame) -> Dict:
        """Return top 3 + bottom 2 products with composite scoring"""
        if len(category_data) < 1:
            return {'top_picks': [], 'avoid_products': []}

        if len(category_data) < 3:
            return self._handle_small_categories(category_data)

        products_with_scores = self._calculate_product_scores(category_data)

        sorted_products = sorted(
            products_with_scores, key=lambda x: x['composite_score'], reverse=True)
        top_picks = sorted_products[:3]
        avoid_products = self._get_avoid_products(
            products_with_scores, sorted_products)

        return {'top_picks': top_picks, 'avoid_products': avoid_products}

    def _calculate_product_scores(self, category_data: pd.DataFrame) -> List[Dict]:
        """Calculate composite scores for products"""
        products_with_scores = []

        for _, product in category_data.iterrows():
            pos_count = product.get('positive_review_count', 0)
            neg_count = product.get('negative_review_count', 0)
            neu_count = product.get('neutral_review_count', 0)
            total_reviews = pos_count + neg_count + neu_count

            review_counts = {
                'positive': pos_count,
                'negative': neg_count,
                'neutral': neu_count,
                'total': total_reviews
            }
            avg_rating = self._calculate_avg_rating(product, review_counts)
            sentiment_score = pos_count / total_reviews if total_reviews > 0 else 0
            volume_score = min(total_reviews / 100.0, 1.0)

            # Calculate sentiment-based avoid score
            pos_percentage = ((pos_count / total_reviews * 100)
                            if total_reviews > 0 else 0)
            neg_percentage = ((neg_count / total_reviews * 100)
                            if total_reviews > 0 else 0)
            neu_percentage = (((total_reviews - pos_count - neg_count) / total_reviews * 100)
                            if total_reviews > 0 else 0)
            sentiment_avoid_score = pos_percentage - neg_percentage - neu_percentage

            composite_score = (
                avg_rating * 0.5 +
                sentiment_score * 5.0 * 0.35 +
                volume_score * 5.0 * 0.15
            )

            products_with_scores.append({
                'product_id': product['product_id'],
                'name': product['name'],
                'avg_rating': avg_rating,
                'review_count': total_reviews,
                'positive_ratio': sentiment_score,
                'composite_score': composite_score,
                'sentiment_avoid_score': sentiment_avoid_score,
                'pos_count': pos_count,
                'neg_count': neg_count
            })

        return products_with_scores

    def _get_avoid_products(
            self, products_with_scores: List[Dict],
            sorted_products: List[Dict]) -> List[Dict]:
        """Get products that should be avoided based on sentiment scores"""
        avoid_products = []
        if len(sorted_products) >= 3:
            poor_sentiment = [p for p in products_with_scores
                            if p['sentiment_avoid_score'] < 65]
            if poor_sentiment:
                by_sentiment = sorted(
                    poor_sentiment, key=lambda x: x['sentiment_avoid_score'])
                avoid_products = by_sentiment[:2]
        return avoid_products


    def generate_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics from aggregated data"""
        total_reviews = (df['positive_review_count'].sum() +
                        df['negative_review_count'].sum() +
                        df['neutral_review_count'].sum())

        sentiment_distribution = {
            'positive': int(df['positive_review_count'].sum()),
            'negative': int(df['negative_review_count'].sum()),
            'neutral': int(df['neutral_review_count'].sum())
        }

        return {
            'total_reviews': int(total_reviews),
            'unique_products': int(df['name'].nunique()),
            'categories_covered': int(df['meta_category'].nunique()),
            'sentiment_distribution': {k: int(v)
                                     for k, v in sentiment_distribution.items()},
            'average_rating': (float(df['avg_rating'].mean())
                             if 'avg_rating' in df.columns else 0.0)
        }
