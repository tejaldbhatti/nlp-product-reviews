"""
This script performs product category clustering on Amazon review data.
It covers data preprocessing, text embedding, various clustering algorithms,
dimensionality reduction for visualization, and final output generation.
"""

# --- Standard Library Imports ---
import sys # Used for sys.exit for clean script exits

# --- Third-Party Library Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan
import umap.umap_ as umap


# Step 1: Data Loading and Initial Inspection
# Loads the raw dataset and performs an initial check for missing values.

try:
    df = pd.read_csv("/content/1429_1.csv", low_memory=False)
except FileNotFoundError:
    print("Error: '1429_1.csv' not found. Please ensure the file exists in /content/.")
    sys.exit(1) # Exit the script if the data is not found.
except (PermissionError, UnicodeDecodeError, pd.errors.EmptyDataError) as e:
    print(f"An unexpected error occurred while loading the data: {e}")
    sys.exit(1)

# Print the shape of the loaded DataFrame.
print(f"Dataset shape: {df.shape}")

# Check for missing values in crucial columns ('name' and 'categories').
print("Missing values in 'name' and 'categories':")
print(df[['name', 'categories']].isnull().sum())

# Drop rows with missing 'name' or 'categories'.
df = df.dropna(subset=['name', 'categories'])

# Remove duplicate product entries based on their 'name'.
df = df.drop_duplicates(subset='name').reset_index(drop=True)

# Convert the 'categories' string (e.g., "Electronics,Tablets") into a list of strings.
# Handles non-string data by converting safely to an empty list.
df['categories'] = df['categories'].apply(
    lambda x: x.split(',') if isinstance(x, str) else []
)

def clean_text(row):
    """
    Combines the product name and its most specific category into a single string.

    This function extracts the product name and the last element of its categories list
    (considered the most specific category), concatenating them into a single string.
    It gracefully handles cases where the categories list might be empty.

    Args:
        row (pd.Series): A row from the DataFrame containing 'name' and 'categories' lists.

    Returns:
        str: The combined string "product_name most_specific_category".
    """
    categories = row['categories']
    # If the categories list is not empty, take the last element; otherwise, use an empty string.
    most_specific_cat = categories[-1] if categories else ""
    # Format the combined string using an f-string.
    return f"{row['name']} {most_specific_cat}"

# Create the 'text_for_clustering' column by applying the clean_text function row-wise.
# This column will serve as the primary text input for generating product embeddings.
df['text_for_clustering'] = df.apply(clean_text, axis=1)

# Display a confirmation message and the first few rows of the updated DataFrame
# to show the newly created 'text_for_clustering' column.
print("\n--- Data Preprocessing Complete (text_for_clustering created) ---")
print(df[['name', 'categories', 'text_for_clustering']].head())


# Step 2: Prepare DataFrame for Embedding and Generate Sentence Embeddings
# Selects relevant columns, performs final cleaning, and generates embeddings.

# Select relevant columns ('id', 'name', 'text_for_clustering') for embedding.
# The 'id' column is crucial for linking back to original data and for final outputs.
# Using .copy() explicitly prevents SettingWithCopyWarning for subsequent modifications.
df_new = df[['id', 'name', 'text_for_clustering']].copy()

# Drop any rows where 'name' or 'text_for_clustering' might still be missing in this new DataFrame.
# This acts as a final safeguard for the input to the embedding model.
df_new = df_new.dropna(subset=['name', 'text_for_clustering'])

# Remove any remaining duplicate products based on 'name' after all text preparation.
# This ensures that the final dataset passed to the embedding model is unique per product.
df_new = df_new.drop_duplicates(subset='name').reset_index(drop=True)

print("\n--- DataFrame Prepared for Embedding (df_new created) ---")
print(f"Number of unique products for embedding: {len(df_new)}")
print("\nFirst 5 rows of df_new with 'text_for_clustering':")
print(df_new.head())

# Load pretrained Sentence Transformer model.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings from the 'text_for_clustering' column of df_new.
print("\nGenerating embeddings...")
embeddings = model.encode(df_new['text_for_clustering'].tolist(), show_progress_bar=True)

print(f"\nEmbeddings generated. Shape: {embeddings.shape}")


# Step 3: Elbow Method and Silhouette Score Analysis (for KMeans)
# Determines an appropriate number of clusters (k) for KMeans.

SSE = [] # Sum of Squared Errors for each k
SILHOUETTE_SCORES = [] # Silhouette Scores for each k
K_RANGE = range(2, 10) # Range of k values to test

for k in K_RANGE:
    # Initialize and fit KMeans model. n_init=10 recommended for sklearn >= 1.4.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    SSE.append(kmeans.inertia_)
    SILHOUETTE_SCORES.append(silhouette_score(embeddings, labels))

# Plot the Elbow Method results to identify optimal k.
plt.figure(figsize=(10, 5))
plt.plot(K_RANGE, SSE, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.show()

# Plot the Silhouette Score results.
plt.figure(figsize=(10, 5))
plt.plot(K_RANGE, SILHOUETTE_SCORES, 'gx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.grid(True)
plt.show()

print("\n--- Elbow Method and Silhouette Score Analysis Complete ---")


# Step 4: Clustering Algorithms Application (KMeans, Agglomerative, HDBSCAN)
# Applies the selected clustering algorithms to the product embeddings.

# Set the optimal number of clusters based on previous analysis.
OPTIMAL_K = 5

# 1) KMeans Clustering
# Fits KMeans and assigns cluster labels to df_new.
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_new['kmeans_cluster'] = kmeans.fit_predict(embeddings)

# Calculates and prints KMeans Silhouette Score.
kmeans_silhouette = silhouette_score(embeddings, df_new['kmeans_cluster'])
print(f"\nKMeans: Number of clusters = {len(set(df_new['kmeans_cluster']))}")
print(
    f"KMeans Silhouette Score = {kmeans_silhouette:.4f}" # Pylint Fix: C0301
)

# 2) Agglomerative Clustering
# Fits Agglomerative Clustering and assigns cluster labels to df_new.
agglo = AgglomerativeClustering(n_clusters=OPTIMAL_K)
df_new['agglo_cluster'] = agglo.fit_predict(embeddings)

# Calculates and prints Agglomerative Silhouette Score.
agglo_silhouette = silhouette_score(embeddings, df_new['agglo_cluster'])
print(f"\nAgglomerative Clustering: Number of clusters = "
      f"{len(set(df_new['agglo_cluster']))}") # Pylint Fix: C0301
print(
    f"Agglomerative Silhouette Score = {agglo_silhouette:.4f}" # Pylint Fix: C0301
)

# 3) HDBSCAN Clustering
# Fits HDBSCAN and assigns cluster labels to df_new.
hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
df_new['hdbscan_cluster'] = hdb.fit_predict(embeddings)

# Calculate HDBSCAN cluster count and noise points.
num_hdb_clusters = len(set(df_new['hdbscan_cluster'])) - \
                   (1 if -1 in df_new['hdbscan_cluster'].values else 0)
num_noise_points = (df_new['hdbscan_cluster'] == -1).sum()

print(f"\nHDBSCAN: Number of clusters = {num_hdb_clusters}")
print(f"HDBSCAN: Number of noise points = {num_noise_points}")

# Calculate Silhouette Score for HDBSCAN, excluding noise points.
if num_hdb_clusters > 1:
    hdb_silhouette = silhouette_score(
        embeddings[df_new['hdbscan_cluster'] != -1],
        df_new['hdbscan_cluster'][df_new['hdbscan_cluster'] != -1]
    )
    print(
        f"HDBSCAN Silhouette Score = {hdb_silhouette:.4f}" # Pylint Fix: C0301
    )
else:
    print("HDBSCAN Silhouette Score: Not applicable (only one cluster or all noise)")

print("\n--- Clustering Algorithms Applied ---")


# Step 5: Visualization of Clusters using UMAP
# UMAP reduces high-dimensional embeddings to 2D for plotting, aiding cluster interpretation.

# UMAP dimensionality reduction configuration.
# n_neighbors: balances local vs. global structure.
# min_dist: controls how tightly points are packed.
UMAP_REDUCER = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
EMBEDDING_2D = UMAP_REDUCER.fit_transform(embeddings)

# Add 2D UMAP coordinates to df_new for plotting.
df_new['UMAP_1'] = EMBEDDING_2D[:, 0]
df_new['UMAP_2'] = EMBEDDING_2D[:, 1]

# Choose one of the cluster columns to visualize.
CLUSTER_COLUMN_TO_VISUALIZE = 'hdbscan_cluster'

plt.figure(figsize=(12, 8))
# Create a color palette based on unique clusters in the chosen column.
palette = sns.color_palette("tab10", len(df_new[CLUSTER_COLUMN_TO_VISUALIZE].unique()))

# Create a scatter plot of the 2D UMAP embeddings, colored by cluster.
sns.scatterplot(
    x='UMAP_1',
    y='UMAP_2',
    hue=df_new[CLUSTER_COLUMN_TO_VISUALIZE],
    palette=palette,
    data=df_new,
    legend='full',
    s=20,
    alpha=0.7
)

plt.title(f"UMAP projection colored by {CLUSTER_COLUMN_TO_VISUALIZE}")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print(f"\n--- UMAP Visualization for '{CLUSTER_COLUMN_TO_VISUALIZE}' Complete ---")


# Step 6: Cluster Summarization and Sample Inspection
# Defines helper functions to summarize cluster content and assess interpretability.

def clean_product_text(text, max_len=80):
    """
    Cleans and truncates a product text string for display in cluster summaries.

    Removes line breaks, extra commas, and truncates the text to a maximum length
    with an ellipsis, ensuring readability in tables.

    Args:
        text (str): The input product text string.
        max_len (int): The maximum desired length for the cleaned text.

    Returns:
        str: The cleaned and truncated product text.
    """
    text = str(text)
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = text.replace(',,,', '').strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."
    return text

def cluster_summary(dataframe, cluster_col, text_col='text_for_clustering', sample_size=3):
    """
    Generates a summary DataFrame for each cluster, including product count and samples.

    Iterates through each unique cluster, provides the count of products, and samples
    a specified number of product texts to give an idea of the cluster's content.
    Noise points (-1 in HDBSCAN) are excluded.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing clustered products (e.g., df_new).
        cluster_col (str): The name of the column containing cluster assignments.
        text_col (str): The name of the column with text used for clustering.
        sample_size (int): The number of sample product texts to display per cluster.

    Returns:
        pd.DataFrame: A summary DataFrame with 'Cluster', 'Count', and 'Sample Products'.
    """
    summary = []
    clusters = sorted(dataframe[cluster_col].dropna().unique())

    if -1 in clusters:
        clusters.remove(-1)

    for cluster_num in clusters:
        data_for_current_cluster = dataframe[dataframe[cluster_col] == cluster_num]

        num_samples = min(sample_size, len(data_for_current_cluster))

        if num_samples > 0:
            sample_texts = data_for_current_cluster[text_col].sample(
                num_samples, random_state=42
            ).values
            cleaned_samples = [clean_product_text(s) for s in sample_texts]
        else:
            cleaned_samples = []

        summary.append({
            'Cluster': cluster_num,
            'Count': len(data_for_current_cluster),
            'Sample Products': " | ".join(cleaned_samples)
        })
    return pd.DataFrame(summary)

# Example Usage: Generate and print the summary for KMeans clustering.
kmeans_summary = cluster_summary(df_new, 'kmeans_cluster')
print("\n--- KMeans Cluster Summary (Sample Products) ---")
print(kmeans_summary.to_string(index=False))

# Step 7: Final Output Generation - Save Cluster Assignments
# This step saves the DataFrame containing product IDs, names,
# clustering text, and all cluster assignments to a CSV file.

# Select relevant columns for the final output CSV.
FINAL_OUTPUT_COLUMNS = [
    'id',
    'name',
    'text_for_clustering',
    'kmeans_cluster',
    'agglo_cluster',
    'hdbscan_cluster'
]

# Create the final DataFrame for saving.
df_final_clusters_output = df_new[FINAL_OUTPUT_COLUMNS].copy()

# Define the output CSV file name.
OUTPUT_CSV_FILENAME = 'category_cluster_with_id.csv'

# Save the DataFrame to a CSV file.
df_final_clusters_output.to_csv(OUTPUT_CSV_FILENAME, index=False)

print("\n--- Final Cluster Assignments Saved ---")
print(
    f"Product cluster assignments have been successfully saved to " # Pylint Fix: C0301
    f"'{OUTPUT_CSV_FILENAME}'"
)
print(
    f"It contains {len(df_final_clusters_output)} rows and "
    f"{len(df_final_clusters_output.columns)} columns."
)
print("\nFirst 5 rows of the saved cluster assignments:")
print(df_final_clusters_output.head())


# Step 8: Cluster Distribution Verification
# This step loads the saved cluster data and prints the count of products
# assigned to each cluster for each of the clustering methods.

try:
    cluster_data = pd.read_csv("/content/category_cluster_with_id.csv")
    print("\n--- 'category_cluster_with_id.csv' loaded successfully for verification ---")
except FileNotFoundError:
    print("Error: 'category_cluster_with_id.csv' not found. Please ensure the file exists.")
    sys.exit(1)
except (PermissionError, UnicodeDecodeError, pd.errors.EmptyDataError) as e:
    print(f"An unexpected error occurred during CSV loading: {e}")
    sys.exit(1)

# Check the number of products in each KMeans cluster.
if 'kmeans_cluster' in cluster_data.columns:
    print("\n--- Product Count per KMeans Cluster ---")
    print(cluster_data['kmeans_cluster'].value_counts().sort_index())
else:
    print("\n'kmeans_cluster' column not found in the DataFrame.")

# Check the number of products in each Agglomerative cluster.
if 'agglo_cluster' in cluster_data.columns:
    print("\n--- Product Count per Agglomerative Cluster ---")
    print(cluster_data['agglo_cluster'].value_counts().sort_index())
else:
    print("\n'agglo_cluster' column not found in the DataFrame.")

# Check the number of products in each HDBSCAN cluster.
if 'hdbscan_cluster' in cluster_data.columns:
    print("\n--- Product Count per HDBSCAN Cluster ---")
    num_noise_points_hdbscan = (cluster_data['hdbscan_cluster'] == -1).sum()
    print(cluster_data['hdbscan_cluster'].value_counts().sort_index())
    if num_noise_points_hdbscan > 0:
        print(f"(Note: Cluster -1 represents {num_noise_points_hdbscan} noise points in HDBSCAN)")
else:
    print("\n'hdbscan_cluster' column not found in the DataFrame.")

print("\n--- Cluster Distribution Verification Complete ---")

# Pylint Fix: C0304 - Final newline missing
