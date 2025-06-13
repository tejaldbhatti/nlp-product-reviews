"""
This script performs sentiment analysis on Amazon product reviews,
including data loading, preprocessing, VADER analysis,
SVC model training, and evaluation.
It generates a CSV file with predicted sentiments and confidence scores.
"""

# --- Standard Library Imports ---
import sys # For sys.exit to gracefully exit the script on errors

# --- Third-Party Library Imports ---
import pandas as pd # For data manipulation and analysis (DataFrames)
import numpy as np # For numerical operations, specifically np.nan for missing values

import matplotlib.pyplot as plt # For plotting and visualizing data (e.g., Confusion Matrix)

# Pylint Fix: Line too long (C0301) - Breaking long import line.
from vaderSentiment.vaderSentiment import ( # For VADER sentiment analysis
    SentimentIntensityAnalyzer
)

from sklearn.feature_extraction.text import TfidfVectorizer # For converting text to numerical features
from sklearn.model_selection import train_test_split # For splitting data into training/testing sets
from sklearn.metrics import ( # For model evaluation metrics
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay # For plotting confusion matrix
)
from sklearn.svm import SVC # For the Support Vector Classifier model


# Step 1: Data Loading and Initial Inspection
# Loads the raw dataset and performs basic checks.
try:
    df = pd.read_csv("/content/1429_1.csv", low_memory=False)
except FileNotFoundError:
    print("Error: '1429_1.csv' not found. "
          "Please ensure the file exists in /content/.")
    sys.exit(1) # Exit the script if the data is not found.
except Exception as e: # Catching a general exception for unexpected issues.
    print(f"An unexpected error occurred during data loading: {e}")
    sys.exit(1)

# Print the shape of the DataFrame (number of rows, number of columns).
print(f"Dataset shape: {df.shape}")

# Print a list of all column names in the DataFrame.
# Pylint Fix: Line too long (C0301) - Breaking string.
print(f"Columns: {list(df.columns)}")

# Display the first 4 rows of the DataFrame for an initial overview.
print("\nFirst few rows:")
print(df.head(4))


# Step 2: Create Initial Sentiment Labels from Ratings
# Defines a function to convert numerical star ratings into
# categorical sentiment labels (positive, neutral, negative) based on rules.
def rating_to_sentiment(rating):
    """
    Converts a numerical product rating into a categorical sentiment label.

    Ratings of 4 or 5 are 'positive', 3 is 'neutral', and 1 or 2 are 'negative'.

    Args:
        rating (int or float): The numerical rating of a product review.

    Returns:
        str: The corresponding sentiment label ('positive', 'neutral', or 'negative').
    """
    if rating >= 4:
        return 'positive'
    if rating == 3:
        return 'neutral'
    return 'negative' # Ratings of 1 or 2 (Pylint Fix: R1705 - no-else-return)


# Apply the defined function to the 'reviews.rating' column to create
# a new 'sentiment' column in the DataFrame.
df['sentiment'] = df['reviews.rating'].apply(rating_to_sentiment)

print("\n--- Initial sentiment labels created based on ratings ---")
print(df[['reviews.rating', 'sentiment']].head())


# Step 3: VADER Sentiment Analysis and Label Reconciliation
# Applies VADER sentiment analysis to review text, compares its
# output with the initial rating-based sentiment, and determines a final
# sentiment label for model training, prioritizing VADER's output.

# Initialize the VADER Sentiment Intensity Analyzer.
analyzer = SentimentIntensityAnalyzer()

def vader_label(text):
    """
    Applies VADER sentiment analysis to a given text and returns a sentiment label.

    The sentiment is determined based on the compound score:
    - 'positive' if compound >= 0.05
    - 'negative' if compound <= -0.05
    - 'neutral' otherwise (between -0.05 and 0.05 exclusive).

    Args:
        text (str): The input text (e.g., a product review).

    Returns:
        str: The sentiment label ('positive', 'negative', or 'neutral').
    """
    # Ensure text is converted to string to handle potential non-string inputs (e.g., NaN).
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral' # Pylint Fix: R1705 - no-else-return

# Apply the VADER sentiment labeling function to the 'reviews.text' column
# and store the results in a new 'vader_sentiment' column.
df['vader_sentiment'] = df['reviews.text'].apply(vader_label)

# Check for mismatches between VADER's sentiment and the initial rating-based sentiment.
mismatches = df[df['vader_sentiment'] != df['sentiment']]
print(f"\nNumber of label mismatches (VADER vs. Rating-based): {len(mismatches)}")

# Temporarily set pandas display option to show full text for review inspection.
pd.set_option('display.max_colwidth', None)

# Display a sample of reviews with mismatched sentiments.
print("\n--- Sample of Reviews with Mismatched Sentiments (first 3) ---")
display_mismatches = mismatches[[
    'reviews.text',
    'reviews.rating',
    'sentiment',
    'vader_sentiment'
]]
print(display_mismatches.head(3))

# Reset pandas display option to its default to avoid affecting subsequent outputs.
pd.reset_option('display.max_colwidth')

# Assign the 'final_label' for model training, trusting VADER's sentiment for its nuance.
df['final_label'] = df['vader_sentiment']

# Print the distribution of the final sentiment labels to check class balance.
print("\n--- Distribution of final sentiment labels ('final_label') ---")
print(df['final_label'].value_counts())


# Step 4: Feature Extraction using TF-IDF
# Converts raw textual review data into numerical features
# using TF-IDF for machine learning.

# Replace any NaN values in the 'reviews.text' column with empty strings.
df['reviews.text'] = df['reviews.text'].replace(np.nan, '')

# Initialize TfidfVectorizer with a limit of 5000 features.
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer to the review text and transform it into TF-IDF features.
X = vectorizer.fit_transform(df['reviews.text'])

# Assign the 'final_label' column as the target variable (y) for the model.
y = df['final_label']

# Optional: Print shapes to confirm dimensions
# print(f"\nShape of X (TF-IDF features): {X.shape}")
# print(f"Shape of y (target labels): {y.shape}")


# Step 5: Data Splitting
# Splits the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print the shapes of the resulting datasets to verify the split.
print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# Step 6: Model Training (Support Vector Classifier - SVC)
# Initializes and trains the Support Vector Classifier (SVC) model.
svc_model = SVC(kernel='linear', probability=True, random_state=42,
                class_weight='balanced')

print("\nTraining the SVC model...")
svc_model.fit(X_train, y_train)


# Step 7: Model Evaluation
# Evaluates the performance of the trained SVC model on the test data.
y_pred = svc_model.predict(X_test)

# Print overall model evaluation metrics.
print("\nModel Evaluation:")
# Print the accuracy score, formatted to 4 decimal places.
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print the classification report.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 8: Confusion Matrix Visualization
# Calculates and visualizes the confusion matrix.
class_labels = svc_model.classes_ # Get class labels from the trained SVC model.
cm = confusion_matrix(y_test, y_pred, labels=class_labels) # Calculate confusion matrix.

# Create a ConfusionMatrixDisplay object for visualization.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

# Plot the confusion matrix.
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix with Sentiment Names')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Step 9: Generate Output CSV with Sentiment and Confidence
# Applies the trained SVC model to the *entire* dataset to predict sentiment
# and confidence scores for all reviews, then exports to a CSV file.

# 1. Get predicted sentiment for the entire dataset using the trained SVC model.
all_predicted_sentiment = svc_model.predict(X)

# 2. Get probabilities for the entire dataset.
all_probabilities = svc_model.predict_proba(X)

# 3. Determine the confidence score for each prediction.
confidence_scores = []
class_labels_model = svc_model.classes_ # Get the order of classes from the model.

for i, pred_sentiment in enumerate(all_predicted_sentiment):
    predicted_class_index = list(class_labels_model).index(pred_sentiment)
    confidence = all_probabilities[i, predicted_class_index]
    confidence_scores.append(confidence)

# 4. Create the final DataFrame for export.
results_df = pd.DataFrame({
    'product_id': df['id'],
    'reviews.text': df['reviews.text'],
    'rating': df['reviews.rating'],
    'original_sentiment_from_rating': df['sentiment'],
    'predicted_sentiment_SVC': all_predicted_sentiment,
    'prediction_confidence': confidence_scores
})

# 5. Export to CSV.
OUTPUT_FILENAME = 'product_reviews_sentiment_and_confidence.csv'
results_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\nSuccessfully generated '{OUTPUT_FILENAME}' with sentiment and confidence scores.")
# Pylint Fix: Breaking line (C0301)
print(f"Preview of the generated CSV file (first 5 rows):\n{results_df.head()}")
