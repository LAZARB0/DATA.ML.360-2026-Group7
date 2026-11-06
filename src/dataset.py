import argparse
from pathlib import Path

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

Path("results").mkdir(exist_ok=True)

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

num_bins = 50

plt.figure(figsize=(10, 5))

plt.hist(
    user_interactions,
    bins=num_bins,
    edgecolor="black"
)

plt.xlabel("Number of Ratings per User")
plt.ylabel("Number of Users")
plt.title("Long-Tail Distribution of User Activity")

plt.tight_layout()
plt.savefig("results/user_activity.png")

# =====================
# Interactions per item
# =====================
item_interactions = ratings.groupby("MovieID").size()

print("\n===== ITEM INTERACTIONS =====")
print(item_interactions.describe())

# Long-tail plot: items
plt.figure(figsize=(10, 5))

item_interactions.sort_values(
    ascending=False
).reset_index(drop=True)

num_bins = 50

plt.hist(
    user_interactions,
    bins=num_bins,
    edgecolor="black"
)

plt.xlabel("Number of Ratings per Item")
plt.ylabel("Number of Items")
plt.title("Long-Tail Distribution of Item Activity")

plt.tight_layout()
plt.savefig("results/item_activity.png")

# ===================
# Rating distribution
# ===================
rating_counts = ratings["Rating"].value_counts().sort_index()

print("\n===== RATING DISTRIBUTION =====")
print(rating_counts)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=ratings,
    x="Rating",
    color="steelblue"
)

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("results/rating_distribution.png")

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

plt.figure(figsize=(12, 5))

plt.bar(
    monthly_activity.index,
    monthly_activity.values,
    width=20
)

plt.title("Rating Activity Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Ratings")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("results/rating_activity.png")

# Temporal train/validation/test split
ratings = ratings.sort_values("Timestamp")
ratings = ratings.reset_index(drop=True)

n = len(ratings)

train_end = int(0.80 * n)
val_end = int(0.90 * n)

train = ratings.iloc[:train_end]
validation = ratings.iloc[train_end:val_end]
test = ratings.iloc[val_end:]

#Report
print("\n===== TEMPORAL SPLIT =====")
print("Train size:", len(train))
print("Validation size:", len(validation))
print("Test size:", len(test))

print("\nTrain period:")
print(train["Datetime"].min(), "->", train["Datetime"].max())

print("\nValidation period:")
print(validation["Datetime"].min(), "->",
      validation["Datetime"].max())

print("\nTest period:")
print(test["Datetime"].min(), "->",
      test["Datetime"].max())
