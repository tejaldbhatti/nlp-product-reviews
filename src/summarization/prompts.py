"""
Few-Shot Prompt Generation for Product Comparisons
"""


def few_shot_comparison_prompt(products: list, category: str, sample_reviews: dict = None) -> str:
    """Few-shot prompting with examples for better product comparison quality"""

    # Build product summary with review quotes
    product_sections = []
    for p in products:
        product_section = (f"- {p['name']}: {p['avg_rating']}/5 rating, "
                           f"{p['review_count']} reviews, "
                           f"{p['positive_ratio']*100:.0f}% positive")

        # Add sample reviews if available
        if sample_reviews and p['name'] in sample_reviews:
            reviews = sample_reviews[p['name']]

            # Get positive reviews
            positive_reviews = [r for r in reviews if r['sentiment'] == 'positive'][:2]
            # Get negative/neutral reviews
            critical_reviews = [r for r in reviews
                               if r['sentiment'] in ['negative', 'neutral']][:2]

            if positive_reviews or critical_reviews:
                product_section += "\n  Customer Feedback:"

                for review in positive_reviews:
                    truncated = ('...' if len(review['text']) > 150 else '')
                    product_section += (f"\n    ✓ \"{review['text'][:150]}{truncated}\" "
                                      f"({review['rating']}/5)")

                for review in critical_reviews:
                    truncated = ('...' if len(review['text']) > 150 else '')
                    product_section += (f"\n    ⚠ \"{review['text'][:150]}{truncated}\" "
                                      f"({review['rating']}/5)")

        product_sections.append(product_section)

    product_info = "\n\n".join(product_sections)

    return f"""Create a product recommendation guide for {category} \
based on customer reviews and ratings.

Products and Customer Reviews:
{product_info}

Use the customer quotes and data above to create a professional buying guide with the following structure:

## Overview
Brief summary with review count and average rating

## Customer Sentiment Analysis
Specific percentages of positive/negative feedback

## What Customers Love
Key strengths mentioned in positive reviews (use actual quotes)

## Common Concerns
Issues raised in negative reviews (use actual quotes)

## Bottom Line
Final recommendation for who should buy and why, based on customer experiences

Ground every statement in actual customer quotes and data provided above."""
