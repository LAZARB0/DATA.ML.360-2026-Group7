import argparse
from collections import defaultdict
from math import log2

import numpy as np
import pandas as pd

from load_data import load_data

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
def build_markov_model(data, min_count=1):

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

        for i in range(len(sequence) - 1):

            current_item = sequence[i]
            next_item = sequence[i + 1]

            transition_counts[current_item][next_item] += 1

    transition_probs = {}

    for item, transitions in transition_counts.items():

        filtered = {
            nxt: cnt
            for nxt, cnt in transitions.items()
            if cnt >= min_count
        }

        if not filtered:
            continue

        total = sum(filtered.values())

        transition_probs[item] = {
            nxt: cnt / total
            for nxt, cnt in filtered.items()
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

    if len(precisions) == 0:
        return {
            "Precision@10": 0.0,
            "Recall@10": 0.0,
            "NDCG@10": 0.0,
            "Coverage": 0.0,
            "Evaluated users": 0,
            "Skipped users": heldout["UserID"].nunique(),
            "UserNDCGs": []
        }

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
        "UserNDCGs": ndcgs,
    }

def bootstrap_ci(scores, n_boot=1000):

    means = []

    scores = np.array(scores)

    for _ in range(n_boot):

        sample = np.random.choice(
            scores,
            size=len(scores),
            replace=True
        )

        means.append(sample.mean())

    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)

    return lower, upper

catalogue_size = (
    movies["MovieID"]
    .nunique()
)

# ==================================================
# Hyperparameter tuning (validation only)
# ==================================================

K_VALUES = [5, 10, 20]
MIN_COUNTS = [1, 3, 5]

tuning_rows = []

for k in K_VALUES:
    for min_count in MIN_COUNTS:

        transition_probs = build_markov_model(
            train,
            min_count=min_count
        )

        popular_items = (
            train["MovieID"]
            .value_counts()
            .index
            .tolist()
        )

        result = evaluate_markov(
            train,
            validation,
            catalogue_size,
            k=k
        )

        result["K"] = k
        result["MinCount"] = min_count

        tuning_rows.append(result)

tuning_results = pd.DataFrame(tuning_rows)

print("\n===== HYPERPARAMETER TUNING =====")
print(
    tuning_results[
        ["K", "MinCount",
         "Precision@10",
         "Recall@10",
         "NDCG@10"]
    ]
    .sort_values("NDCG@10", ascending=False)
    .round(4)
    .to_string(index=False)
)

best_row = tuning_results.loc[
    tuning_results["NDCG@10"].idxmax()
]

best_k = int(best_row["K"])
best_min_count = int(best_row["MinCount"])

print("\nBest parameters")
print(f"K = {best_k}")
print(f"MinCount = {best_min_count}")


# ==================================================
# Final model (train + validation)
# ==================================================

fit_data = pd.concat(
    [train, validation],
    ignore_index=True
)

transition_probs = build_markov_model(
    fit_data,
    min_count=best_min_count
)

popular_items = (
    fit_data["MovieID"]
    .value_counts()
    .index
    .tolist()
)

final_result = evaluate_markov(
    fit_data,
    test,
    catalogue_size,
    k=best_k
)

print("\n===== FINAL TEST RESULTS =====")

print(
    pd.DataFrame([final_result])
    .drop(columns=["UserNDCGs"])
    .round(4)
    .to_string(index=False)
)


# ==================================================
# Confidence interval
# ==================================================
if len(final_result["UserNDCGs"]) > 0:
    lower, upper = bootstrap_ci(
        final_result["UserNDCGs"]
    )

print(
    f"\nNDCG@10 = {final_result['NDCG@10']:.6f}"
)

print(
    f"95% CI = [{lower:.6f}, {upper:.6f}]"
)


# ==================================================
# User activity breakdown
# ==================================================

user_activity = train.groupby(
    "UserID"
).size()

median_activity = user_activity.median()

low_users = set(
    user_activity[
        user_activity <= median_activity
    ].index
)

high_users = set(
    user_activity[
        user_activity > median_activity
    ].index
)

test_low = test[
    test["UserID"].isin(low_users)
]

test_high = test[
    test["UserID"].isin(high_users)
]

low_result = evaluate_markov(
    fit_data,
    test_low,
    catalogue_size,
    k=best_k
)

high_result = evaluate_markov(
    fit_data,
    test_high,
    catalogue_size,
    k=best_k
)


# ==================================================
# Early / Late test period breakdown
# ==================================================

mid_timestamp = test["Timestamp"].median()

test_early = test[
    test["Timestamp"] <= mid_timestamp
]

test_late = test[
    test["Timestamp"] > mid_timestamp
]

early_result = evaluate_markov(
    fit_data,
    test_early,
    catalogue_size,
    k=best_k
)

late_result = evaluate_markov(
    fit_data,
    test_late,
    catalogue_size,
    k=best_k
)


# ==================================================
# Disaggregated results table
# ==================================================

disaggregated = pd.DataFrame([

    {
        "Model": "Markov Chain",
        "Breakdown": "User activity",
        "Group": "Low",
        "Definition": f"<= {median_activity:.0f} train ratings",
        "Precision@10": low_result["Precision@10"],
        "Recall@10": low_result["Recall@10"],
        "NDCG@10": low_result["NDCG@10"],
        "Coverage": low_result["Coverage"],
        "Evaluated users": low_result["Evaluated users"],
    },

    {
        "Model": "Markov Chain",
        "Breakdown": "User activity",
        "Group": "High",
        "Definition": f"> {median_activity:.0f} train ratings",
        "Precision@10": high_result["Precision@10"],
        "Recall@10": high_result["Recall@10"],
        "NDCG@10": high_result["NDCG@10"],
        "Coverage": high_result["Coverage"],
        "Evaluated users": high_result["Evaluated users"],
    },

    {
        "Model": "Markov Chain",
        "Breakdown": "Test period",
        "Group": "Early",
        "Definition": "First half of test period",
        "Precision@10": early_result["Precision@10"],
        "Recall@10": early_result["Recall@10"],
        "NDCG@10": early_result["NDCG@10"],
        "Coverage": early_result["Coverage"],
        "Evaluated users": early_result["Evaluated users"]
    },

    {
        "Model": "Markov Chain",
        "Breakdown": "Test period",
        "Group": "Late",
        "Definition": "Second half of test period",
        "Precision@10": late_result["Precision@10"],
        "Recall@10": late_result["Recall@10"],
        "NDCG@10": late_result["NDCG@10"],
        "Coverage": late_result["Coverage"],
        "Evaluated users": late_result["Evaluated users"]
    }
])

print("\n===== DISAGGREGATED RESULTS =====")

print(
    disaggregated
    .round(4)
    .to_string(index=False)
)


# ==================================================
# Sparsity ablation
# ==================================================

ablation_rows = []

for fraction, train_subset in [
    ("100%", train),
    ("50%", train.sample(frac=0.5, random_state=42)),
    ("20%", train.sample(frac=0.2, random_state=42))
]:

    transition_probs = build_markov_model(
        train_subset,
        min_count=best_min_count
    )

    popular_items = (
        train_subset["MovieID"]
        .value_counts()
        .index
        .tolist()
    )

    result = evaluate_markov(
        train_subset,
        validation,
        catalogue_size,
        k=best_k
    )

    ablation_rows.append({
        "Training Fraction": fraction,
        "Precision@10": result["Precision@10"],
        "Recall@10": result["Recall@10"],
        "NDCG@10": result["NDCG@10"],
        "Coverage": result["Coverage"],
        "Evaluated users": result["Evaluated users"],
    })

ablation_results = pd.DataFrame(
    ablation_rows
)

print("\n===== SPARSITY ABLATION =====")

print(
    ablation_results
    .round(4)
    .to_string(index=False)
)