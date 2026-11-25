# Zomato Restaurant Analysis & Clustering

An end-to-end **Exploratory Data Analysis (EDA) and unsupervised machine learning** project built in Python using Jupyter Notebook and Google Colab. It combines Zomato restaurant metadata with customer reviews to study restaurant characteristics, analyse review patterns, and group reviewed restaurants into data-driven segments.

## Project Objectives

- Clean and merge the restaurant metadata and review datasets.
- Explore ratings, estimated cost, cuisines, collections, reviewer activity, and review sentiment.
- Test selected relationships in the data using non-parametric statistical tests.
- Compare **K-Means**, **Agglomerative Clustering**, and **Gaussian Mixture Models (GMM)**.
- Interpret restaurant clusters and export a reusable clustering pipeline.

## Datasets

The notebook uses two CSV files:

| File | Description | Raw records |
|---|---|---:|
| `Zomato Restaurant names and Metadata.csv` | Restaurant name, Zomato link, estimated cost for two, collections, cuisines, and timings | 105 |
| `Zomato Restaurant reviews.csv` | Restaurant reviews, ratings, reviewer details, timestamps, and picture counts | 10,000 |

The metadata and reviews are joined using a **left one-to-many merge** on restaurant name. After removing 36 exact duplicate review rows, the analysis retains **9,964 reviews**. Review data is available for 100 of the 105 restaurants; the remaining five are retained in the catalogue but excluded from review-based clustering.

## Workflow

1. **Data inspection and cleaning:** inspect schema, missing values, duplicates, data types, and unique values; parse cost and rating fields and handle invalid ratings as missing.
2. **Merge and aggregation:** join the two datasets, then create restaurant-level measures such as average rating, rating spread, review count, positive/negative review shares, sentiment, picture activity, and reviewer statistics.
3. **EDA and hypothesis testing:** visualise rating, cost, cuisine, collection, and review patterns; use Spearman correlation and the Mann–Whitney U test for selected hypotheses.
4. **Text and feature engineering:** clean review text, calculate sentiment polarity, generate TF-IDF features, reduce review text with Truncated SVD, encode the ten most frequent cuisines, and combine these with numerical restaurant and reviewer features.
5. **Preprocessing:** median-impute missing numeric values, standardise the feature matrix, and apply PCA. The executed run reduces 28 features to 14 components while retaining approximately **90.63%** of the variance.
6. **Clustering and evaluation:** tune and compare K-Means, Agglomerative Clustering, and GMM using Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Score, cluster sizes, and repeated 80% subsampling stability (Adjusted Rand Index).
7. **Export:** save the selected K-Means pipeline, model comparison, cluster profiles, and restaurant-level cluster assignments.

## Selected Findings

- The listed restaurant cost estimate and average rating have a **moderate positive association** (Spearman ρ = 0.3826, p = 0.000085).
- Cuisine count is **not significantly associated** with average rating in this sample (Spearman ρ = 0.0943, p = 0.3507).
- Reviews with pictures have a higher mean rating than reviews without pictures (3.863 vs. 3.536); the difference is statistically significant in the Mann–Whitney U test (p = 0.0026). Both groups have a median rating of 4.

These are associations in the supplied dataset and should not be interpreted as causal effects.

## Clustering Results

The executed notebook output selects **K-Means with 4 clusters** as the practical final model. GMM has the highest Silhouette Score, while K-Means has the strongest subsampling stability in the recorded comparison and supports direct assignment through learned centroids.

| Model | Clusters | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Stability ARI ↑ |
|---|---:|---:|---:|---:|---:|
| K-Means | 4 | 0.1750 | 1.7414 | 19.3715 | **0.7762** |
| Agglomerative (Ward) | 4 | 0.1615 | **1.6111** | 16.5772 | 0.6183 |
| Gaussian Mixture (Full) | 7 | **0.1799** | 1.7696 | 15.3996 | 0.4911 |

The selected K-Means cluster profiles are summarised below. Cluster IDs are model labels, not ordinal rankings.

| Cluster | Restaurants | Mean estimated cost (₹) | Mean rating | Mean sentiment |
|---:|---:|---:|---:|---:|
| 0 | 16 | 515.62 | 3.64 | 0.25 |
| 1 | 37 | 1,097.30 | 3.59 | 0.25 |
| 2 | 32 | 515.62 | 3.21 | 0.17 |
| 3 | 15 | 1,456.67 | 4.45 | 0.43 |

## Tools & Libraries

- **Environment:** Google Colab / Jupyter Notebook
- **Data analysis:** Python, Pandas, NumPy, SciPy
- **Visualisation:** Matplotlib, Seaborn
- **Text processing:** TextBlob, NLTK, scikit-learn TF-IDF
- **Machine learning:** scikit-learn (preprocessing, PCA, K-Means, Agglomerative Clustering, Gaussian Mixture Models, evaluation metrics)
- **Model export:** Joblib

## Run in Google Colab

1. Open [`Zomato_Project.ipynb`](Zomato_Project.ipynb) in the repository and select **Open in Colab**, or upload the notebook to Google Colab.
2. Upload both CSV files to the Colab working directory (`/content`) or place them in a `data/` folder.
3. Run the notebook cells from top to bottom. The notebook searches for the CSV files by their exact filenames.
4. Generated CSV reports and the saved model are written to an `outputs/` directory.

## Generated Outputs

The notebook exports the following files under `outputs/`:

- `zomato_merged_cleaned_reviews.csv` — merged review-level dataset.
- `zomato_restaurant_level_features.csv` — restaurant-level analytical features.
- `zomato_model_comparison.csv` — tuned clustering metrics.
- `zomato_kmeans_clustering_pipeline.joblib` — fitted preprocessing and K-Means pipeline.
- `zomato_kmeans_cluster_profiles.csv` — summary statistics for each cluster.
- `zomato_reviewed_restaurant_segments.csv` — reviewed restaurants with cluster assignments.
- `zomato_restaurant_cluster_assignments.csv` — full restaurant catalogue with cluster status.

## Limitations

This analysis uses a small, supplied Hyderabad restaurant snapshot. Review sentiment is an automated text-based estimate, clusters are exploratory rather than ground-truth categories, and five restaurants without review data are not assigned to a cluster. The datasets do not provide verified revenue, transaction volume, delivery performance, or current customer behaviour; the findings should therefore be validated before operational use.
