import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

plt.style.use("seaborn-v0_8")

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="ml-1m")
args = parser.parse_args()

ratings, users, movies = load_data(args.data_dir)
"""
# ==============================
# Basic dataset characterization
# ==============================
n_users = ratings["UserID"].nunique()
n_items = ratings["MovieID"].nunique()
n_interactions = len(ratings)

density = n_interactions / (n_users * n_items)

print("===== DATASET SUMMARY =====")
print(f"Users: {n_users}")
print(f"Items: {n_items}")
print(f"Interactions: {n_interactions}")
print(f"Density: {density:.4%}")

# =====================
# Interactions per user
# =====================
user_interactions = ratings.groupby("UserID").size()

print("\n===== USER INTERACTIONS =====")
print(user_interactions.describe())

# Long-tail plot: users

user_interactions.sort_values(
    ascending=False
).reset_index(drop=True).plot()



# =====================
# Interactions per item
# =====================
item_interactions = ratings.groupby("MovieID").size()

print("\n===== ITEM INTERACTIONS =====")
print(item_interactions.describe())

item_interactions.sort_values(
    ascending=False
).reset_index(drop=True).plot()


# ===================
# Rating distribution
# ===================
rating_counts = ratings["Rating"].value_counts().sort_index()

print("\n===== RATING DISTRIBUTION =====")
print(rating_counts)


#==================
# Timespan analysis
#==================
ratings["Datetime"] = pd.to_datetime(
    ratings["Timestamp"],
    unit="s"
)

start_date = ratings["Datetime"].min()
end_date = ratings["Datetime"].max()

print("\n===== TIME SPAN =====")
print("Start:", start_date)
print("End:  ", end_date)
print("Duration:", end_date - start_date)

# Activity over time
monthly_activity = (
    ratings
    .set_index("Datetime")
    .resample("M")
    .size()
)
"""

# Temporal train/validation/test split
ratings = ratings.sort_values("Timestamp")
ratings = ratings.reset_index(drop=True)

n = len(ratings)

train_end = int(0.80 * n)
val_end = int(0.90 * n)

train = ratings.iloc[:train_end]
validation = ratings.iloc[train_end:val_end]
test = ratings.iloc[val_end:]

####################################################
# Most popular

import math

K = 10
POSITIVE_THRESHOLD = 4  


# Metrics for one user
def precision_at_k(recommended, relevant, k):
    # Precision@k = relevant items in top-k / k
    asd = sum(1 for movie in recommended[:k] if movie in relevant)
    return asd / k


def recall_at_k(recommended, relevant, k):
    # Recall@k = relevant items in top-k / total relevant items for the user
    # users with no relevant items are skipped in the evaluation loop
    asd = sum(1 for movie in recommended[:k] if movie in relevant)
    return asd / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    # DCG = sum of 1/log2(position+1) over the hits in the top k
    # ideal = DCG of the best possible list (min(#relevant, k) hits at the top)
    dcg, ideal = 0, 0
    for index, movie in enumerate(recommended[:k]):
        if movie in relevant:
            dcg += 1 / math.log2(index + 2)
    for index in range(min(len(relevant), k)):
        ideal += 1 / math.log2(index + 2)
    return dcg / ideal


def catalogue_coverage(all_recommendations, catalogue_size):
    # distinct movies recommended to at least one user / movies in the catalogue
    unique = set()
    for i in all_recommendations:
        unique.update(i)
    return len(unique) / catalogue_size


# Popularity rankings full list, most popular first
def most_popular_rankings(data):
    counts = data.value_counts("MovieID")

    # Variant 1: all-time, count all ratings
    ranking_count = counts.index.tolist()

    # Variant 2: all-time, count only positive ratings (>= 4)
    positive = data[data["Rating"] >= POSITIVE_THRESHOLD]
    ranking_positive = positive.value_counts("MovieID").index.tolist()

    # Variant 3: average rating, only movies with at least the mean number of ratings
    mean_count = counts.mean()
    avg = data.groupby("MovieID")["Rating"].mean()
    avg_mean = avg[counts.ge(mean_count)]
    ranking_average = avg_mean.sort_values(ascending=False).index.tolist()

    # TODO: Remove prints
    print(f"Mean ratings per movie: {mean_count:.1f}, "
          f"median: {counts.median():.1f}, "
          f"movies passing the mean threshold: {counts.ge(mean_count).sum()}")

    return {
        "count (all ratings)": ranking_count,
        "positive (rating >= 4)": ranking_positive,
        "average (min count = mean)": ranking_average,
    }


# Top-k for one user: skip movies the user has already rated
def recommend(ranking, seen, k=K):
    recommended = []
    for movie in ranking:
        if movie not in seen:
            recommended.append(movie)
            if len(recommended) == k:
                break
    return recommended


# Evaluate one ranking on a held-out set
def evaluate(ranking, fit_data, heldout, catalogue_size, k=K):
    # Movies each user has rated in the fitting data
    seen_by_user = fit_data.groupby("UserID")["MovieID"].apply(set).to_dict()

    # Relevant = held-out movies rated >= 4
    relevant_by_user = (heldout[heldout["Rating"] >= POSITIVE_THRESHOLD].groupby("UserID")["MovieID"].apply(set))

    precisions, recalls, ndcgs = [], [], []
    all_recommendations = []

    for user, relevant in relevant_by_user.items():
        seen = seen_by_user.get(user, set())
        relevant = relevant - seen # already seen movies can't be hits
        if not relevant: 
            continue

        recommended = recommend(ranking, seen, k)
        all_recommendations.append(recommended)

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    n = len(precisions)
    precision = sum(precisions) / n
    recall = sum(recalls) / n
    ndcg = sum(ndcgs) / n
    coverage = catalogue_coverage(all_recommendations, catalogue_size)
    skipped = heldout["UserID"].nunique() - n

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "NDCG@10": ndcg,
        "Coverage": coverage,
        "Evaluated users": n,
        "Skipped users": skipped,
    }


# Run
catalogue_size = movies["MovieID"].nunique() # all movies in movies.dat
rows = []

# Validation: popularity from train only TODO: remove validations only part
rankings = most_popular_rankings(train)
for name, ranking in rankings.items():
    top10_titles = movies.set_index("MovieID").loc[ranking[:10], "Title"].tolist()
    print(f"Top 10 - {name}:") # TODO: Remove movie prints
    for title in top10_titles:
        print(title)
    print("-----------------------------------")
    row = evaluate(ranking, train, validation, catalogue_size)
    row["Split"] = "validation"
    row["Variant"] = name
    rows.append(row)

# Test: popularity from train + validation -> fit on train + validation, evaluated on test
fit_data = pd.concat([train, validation], ignore_index=True)
rankings = most_popular_rankings(fit_data)
for name, ranking in rankings.items():
    row = evaluate(ranking, fit_data, test, catalogue_size)
    row["Split"] = "test"
    row["Variant"] = name
    rows.append(row)

results = pd.DataFrame(rows)
results = results[["Split", 
                    "Variant", 
                    "Precision@10", 
                    "Recall@10",
                    "NDCG@10",
                    "Coverage",
                    #"Evaluated users",
                    #"Skipped users"
                    ]]

pd.set_option("display.width", 200)
print("\nMost popular results")
print(results.round(6).to_string(index=False))
