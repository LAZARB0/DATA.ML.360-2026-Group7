import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")

# Load MovieLens 1M
ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    engine="python",
    names=["UserID", "Gender", "Age", "Occupation", "ZipCode"]
)

movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["MovieID", "Title", "Genres"]
)

# Basic dataset characterization
n_users = ratings["UserID"].nunique()
n_items = ratings["MovieID"].nunique()
n_interactions = len(ratings)

density = n_interactions / (n_users * n_items)

print("===== DATASET SUMMARY =====")
print(f"Users: {n_users}")
print(f"Items: {n_items}")
print(f"Interactions: {n_interactions}")
print(f"Density: {density:.4%}")

# Interactions per user
user_interactions = ratings.groupby("UserID").size()

print("\n===== USER INTERACTIONS =====")
print(user_interactions.describe())

item_interactions = ratings.groupby("MovieID").size()

print("\n===== ITEM INTERACTIONS =====")
print(item_interactions.describe())