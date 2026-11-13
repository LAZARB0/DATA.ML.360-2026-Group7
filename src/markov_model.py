import argparse
from collections import defaultdict
from math import log2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

plt.style.use("seaborn-v0_8")


K = 10
POSITIVE_THRESHOLD = 4


# Data load 
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="ml-1m")
args = parser.parse_args()

ratings, users, movies = load_data(args.data_dir)

# Temporal train/validation/test split
ratings = ratings.sort_values("Timestamp")
ratings = ratings.reset_index(drop=True)

n = len(ratings)

train_end = int(0.80 * n)
val_end = int(0.90 * n)

train = ratings.iloc[:train_end]
validation = ratings.iloc[train_end:val_end]
test = ratings.iloc[val_end:]


# Build Markov model
def build_markov_model(data):

    data = data.sort_values(
        ["UserID", "Timestamp"]
    )

    user_sequences = (
        data.groupby("UserID")["MovieID"]
        .apply(list)
    )

    transition_counts = defaultdict(
        lambda: defaultdict(int)
    )

    for sequence in user_sequences:

        if len(sequence) < 2:
            continue

        for i in range(len(sequence) - 1):

            current_item = sequence[i]
            next_item = sequence[i + 1]

            transition_counts[current_item][next_item] += 1

    transition_probs = {}

    for item, transitions in transition_counts.items():

        total = sum(transitions.values())

        transition_probs[item] = {
            next_item: count / total
            for next_item, count in transitions.items()
        }

    return transition_probs

# Metrics
def precision_at_k(recommended, relevant, k):

    if not recommended:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)

    return hits / k


def recall_at_k(recommended, relevant, k):

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & relevant)

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):

    dcg = 0.0

    for rank, item in enumerate(recommended[:k], start=1):

        if item in relevant:

            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def catalogue_coverage(recommendations, catalogue_size):

    recommended_items = set()

    for recs in recommendations:
        recommended_items.update(recs)

    return len(recommended_items) / catalogue_size

# Validation model
transition_probs = build_markov_model(train)

popular_items = (
    train["MovieID"]
    .value_counts()
    .index
    .tolist()
)

# Recommendation function
def recommend(last_item, k=10):

    if last_item not in transition_probs:

        return popular_items[:k]

    candidates = transition_probs[last_item]

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [item for item, _ in ranked[:k]]

# Evaluation
def evaluate_markov(
    fit_data,
    heldout,
    catalogue_size,
    k=10
):

    history_by_user = (
        fit_data
        .sort_values(["UserID", "Timestamp"])
        .groupby("UserID")["MovieID"]
        .apply(list)
        .to_dict()
    )

    relevant_by_user = (
        heldout[heldout["Rating"] >= POSITIVE_THRESHOLD]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    precisions = []
    recalls = []
    ndcgs = []

    all_recommendations = []

    for user, relevant in relevant_by_user.items():

        history = history_by_user.get(user)

        if history is None or len(history) == 0:
            continue

        last_item = history[-1]

        recommended = recommend(
            last_item,
            k
        )

        all_recommendations.append(
            recommended
        )

        precisions.append(
            precision_at_k(
                recommended,
                relevant,
                k
            )
        )

        recalls.append(
            recall_at_k(
                recommended,
                relevant,
                k
            )
        )

        ndcgs.append(
            ndcg_at_k(
                recommended,
                relevant,
                k
            )
        )

    n = len(precisions)

    return {
        "Precision@10": np.mean(precisions),
        "Recall@10": np.mean(recalls),
        "NDCG@10": np.mean(ndcgs),
        "Coverage": catalogue_coverage(
            all_recommendations,
            catalogue_size
        ),
        "Evaluated users": n,
        "Skipped users":
            heldout["UserID"].nunique() - n,
    }

# Final model: Train + Validation
fit_data = pd.concat(
    [train, validation],
    ignore_index=True
)

transition_probs = build_markov_model(
    fit_data
)

popular_items = (
    fit_data["MovieID"]
    .value_counts()
    .index
    .tolist()
)

catalogue_size = (
    movies["MovieID"]
    .nunique()
)

# Final evaluation on test
row = evaluate_markov(
    fit_data,
    test,
    catalogue_size,
    k=K
)

results = pd.DataFrame([row])

print("\n===== MARKOV CHAIN RESULTS =====")
print(
    results
    .round(6)
    .to_string(index=False)
)