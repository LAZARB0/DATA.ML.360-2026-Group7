import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking   # pip install implicit
from load_data import load_data
import argparse

pd.set_option("display.width", 200)

RUN_TUNING = True # 12 tuning settings on validation + validation breakdowns
RUN_TEST = True # final test run with the selected settings
RUN_SPARSITY = True # sparsity ablation: 100% / 50% / 20% of the training data

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


class HybridRecommender:
    def __init__(self,
                 factors=100,
                 learning_rate=0.01,
                 regularization=0.01,
                 dtype=np.float32,
                 iterations=100,
                 num_threads=1, # 1 thread = reproducible results; 0 = all cores
                 random_state=67,
                 positive_threshold=4,
                 show_progress=True,
                 alpha=0.9 # 1 = only BPR, 0 = only content
                 ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.dtype = dtype
        self.iterations = iterations
        self.num_threads = num_threads
        self.random_state = random_state
        self.positive_threshold = positive_threshold
        self.show_progress = show_progress
        self.alpha = alpha

    def fit(self, train, movies):
        positive_train = train[train["Rating"] >= self.positive_threshold]

        # fallback for cold-start users: movies by number of positive ratings
        self.popular = positive_train.value_counts("MovieID").index.tolist()

        self.user_ids = np.sort(train.UserID.unique()) # all training users
        self.movie_ids = np.sort(movies.MovieID.unique()) # full catalogue

        # content features: genres + release decade, rows in movie_ids order
        indexed = movies.sort_values("MovieID").set_index("MovieID")
        genre_df = indexed["Genres"].str.get_dummies(sep="|")
        years = indexed["Title"].str.extract(r"\((\d{4})\)$", expand=False).astype(int)
        decades = (years // 10) * 10
        decade_dummies = pd.get_dummies(decades, prefix="decade", dtype=int)
        self.genres = genre_df.join(decade_dummies).to_numpy()
        self.movie_length = np.linalg.norm(self.genres, axis=1)

        # user x movie matrix of positive ratings (1 = rated 4 or 5)
        shape = (len(self.user_ids), len(self.movie_ids))
        ui = np.searchsorted(self.user_ids, positive_train.UserID)
        mi = np.searchsorted(self.movie_ids, positive_train.MovieID)
        self.observed = csr_matrix((np.ones(len(positive_train)), (ui, mi)), shape=shape, dtype=self.dtype)

        # user profiles -> number of liked movies per genre / decade
        self.profiles = self.observed @ self.genres

        # movies that user has rated
        self.seen = train.groupby("UserID")["MovieID"].apply(set).to_dict()

        # collaborative filtering part
        self.bpr = BayesianPersonalizedRanking(factors=self.factors,
                                               learning_rate=self.learning_rate,
                                               regularization=self.regularization,
                                               dtype=self.dtype,
                                               iterations=self.iterations,
                                               num_threads=self.num_threads,
                                               random_state=self.random_state)
        self.bpr.fit(self.observed, show_progress=self.show_progress)
        return self

    # row of the user in user_ids, or None if the user is not in the training data
    def _user_row(self, user_id):
        row = np.searchsorted(self.user_ids, user_id)
        if row == len(self.user_ids) or self.user_ids[row] != user_id:
            return None
        return row

    def bpr_scores(self, user_id):
        row = self._user_row(user_id)
        if row is None: # if user is not in the training data
            return np.zeros(len(self.movie_ids)) # returns 0 with every movie in the catalogue
        return np.dot(self.bpr.user_factors[row], self.bpr.item_factors.T)

    def content_scores(self, user_id):
        row = self._user_row(user_id)
        if row is None:
            return np.zeros(len(self.movie_ids))
        profile_length = np.linalg.norm(self.profiles[row]) # length of user's profile vector
        if profile_length == 0: # user has no positive ratings
            return np.zeros(len(self.movie_ids))
        scores = np.dot(self.profiles[row], self.genres.T) 
        return scores / profile_length / self.movie_length # cosine similarity

    def _minmax(self, x):
        minimum, maximum = np.min(x), np.max(x)
        if minimum == maximum:
            return np.zeros_like(x)
        return (x - minimum) / (maximum - minimum)

    def recommend(self, user_id, n=10):
        # cold-start user: fall back to the most popular movies -> if no data from users recommend most popular movies
        if self._user_row(user_id) is None:
            return self.popular[:n]
        # α·CF + (1−α)·content
        combined = (self.alpha * self._minmax(self.bpr_scores(user_id)) + (1 - self.alpha) * self._minmax(self.content_scores(user_id)))

        # if user has already seen movie -> no recommendation for that movie
        seen = list(self.seen.get(user_id, set()))
        combined[np.searchsorted(self.movie_ids, seen)] = -np.inf # set the score of seen movie to -infinity

        top = np.flip(np.argsort(combined))[:n] # sort movies to highest to lowest
        return self.movie_ids[top].tolist()



# Metrics for one user
def precision_at_k(recommended, relevant, k):
    # Precision@k = relevant items in top-k / k
    hits = sum(1 for movie in recommended[:k] if movie in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    # Recall@k = relevant items in top-k / total relevant items for the user
    # (users with no relevant items are skipped in evaluate)
    hits = sum(1 for movie in recommended[:k] if movie in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    # DCG = sum of 1/log2(position+1) over the hits in the top k
    # ideal = DCG of the best possible list (min(#relevant, k) hits at the top)
    dcg, ideal = 0, 0
    for index, movie in enumerate(recommended[:k]):
        if movie in relevant:
            dcg += 1 / np.log2(index + 2)
    for index in range(min(len(relevant), k)):
        ideal += 1 / np.log2(index + 2)
    return dcg / ideal


def catalogue_coverage(all_recommendations, catalogue_size):
    unique = set()
    for recs in all_recommendations:
        unique.update(recs)
    return len(unique) / catalogue_size


def evaluate(model, heldout, k=10):
    relevant_by_user = (heldout[heldout["Rating"] >= model.positive_threshold]
                        .groupby("UserID")["MovieID"].apply(set))
    precisions, recalls, ndcgs = [], [], []
    all_recommendations = []
    per_user = []

    for user, relevant in relevant_by_user.items():
        seen = model.seen.get(user, set())
        relevant = relevant - seen # already seen movies can't be hits
        if not relevant:
            continue

        recommended = model.recommend(user, k)
        all_recommendations.append(recommended)

        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        nd = ndcg_at_k(recommended, relevant, k)

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(nd)
        per_user.append({"UserID": user, "Precision@10": p, "Recall@10": r, "NDCG@10": nd})

    n_users = len(precisions)
    summary = {
        "Precision@10": sum(precisions) / n_users,
        "Recall@10": sum(recalls) / n_users,
        "NDCG@10": sum(ndcgs) / n_users,
        "Coverage": catalogue_coverage(all_recommendations, len(model.movie_ids)),
        "Evaluated users": n_users,
        "Skipped users": heldout["UserID"].nunique() - n_users,
    }
    return summary, pd.DataFrame(per_user)



# disaggregation
METRICS = ["Precision@10", "Recall@10", "NDCG@10"]

def by_user_activity(per_user, fit_data):
    # groups: cold-start (0 training ratings)
    # low (below the median of training users)
    # high (at or above it)
    counts = fit_data.groupby("UserID").size()
    threshold = counts.median()

    n_user_ratings = per_user["UserID"].map(counts).fillna(0)
    group = np.where(n_user_ratings == 0, "cold-start (0)", 
                     np.where(n_user_ratings < threshold, f"low (<{threshold:.0f})", f"high (>={threshold:.0f})"))

    grouped = per_user.assign(group=group).groupby("group")
    table = grouped[METRICS].mean()
    table["Users"] = grouped.size()
    return table


def by_time_period(model, heldout):
    # cut as its median timestamp -> each half is evaluated separately
    cut = heldout["Timestamp"].median()
    halves = {
        "early": heldout[heldout["Timestamp"] < cut],
        "late": heldout[heldout["Timestamp"] >= cut],
    }
    rows = []
    for name, part in halves.items():
        summary, _ = evaluate(model, part)
        rows.append({"period": name, # early or late
                     "from": pd.to_datetime(part["Timestamp"].min(), unit="s").date(), # first day
                     "to": pd.to_datetime(part["Timestamp"].max(), unit="s").date(), # last day
                     **{m: summary[m] for m in METRICS}, # P@10, R@10, NDCG@10
                     "Users": summary["Evaluated users"]}) # users in this half
    return pd.DataFrame(rows).set_index("period")



# tuning on validation models fitted on train -> Budget: 4 BPR x 3 alpha = 12 settings
if RUN_TUNING:
    factors_grid = [50, 100] # latent factors for BRP
    regularization_grid = [0.01, 0.1] # regularization for BRP
    alpha_grid = [0.75, 0.9, 1.0] # weight of BPR vs. content

    rows = []
    for factors, regularization in itertools.product(factors_grid, regularization_grid):
        start = time.time()
        model = HybridRecommender(factors=factors, regularization=regularization,
                                  show_progress=False).fit(train, movies)
        print(f"Fitted factors={factors}, regularization={regularization} "
              f"in {time.time() - start:.0f} s", flush=True)

        # alpha only changes recommend(), so each BPR setting is fitted once
        for alpha in alpha_grid:
            model.alpha = alpha
            summary, _ = evaluate(model, validation)
            rows.append({"factors": factors, "regularization": regularization,
                         "alpha": alpha, **summary})
            print(f"  alpha={alpha}: NDCG@10={summary['NDCG@10']:.4f}", flush=True)

    tuning = pd.DataFrame(rows).sort_values("NDCG@10", ascending=False)
    columns = ["factors", "regularization", "alpha",
               "Precision@10", "Recall@10", "NDCG@10", "Coverage"]
    print(f"\nSettings evaluated: {len(tuning)}")
    print(tuning[columns].round(4).to_string(index=False))

    best = tuning.iloc[0]
    print(f"\nSelected: factors={int(best['factors'])}, "
          f"regularization={best['regularization']}, alpha={best['alpha']} "
          f"(validation NDCG@10 = {best['NDCG@10']:.4f})")

    model = HybridRecommender(factors=int(best["factors"]),
                              regularization=best["regularization"],
                              alpha=best["alpha"],
                              show_progress=False).fit(train, movies)
    summary, per_user = evaluate(model, validation)

    print("\nBy user activity (validation, selected model):")
    print(by_user_activity(per_user, train).round(4).to_string())

    print("\nBy time period (validation, selected model):")
    print(by_time_period(model, validation).round(4).to_string())


BEST_FACTORS = 100 
BEST_REGULARIZATION = 0.1
BEST_ALPHA = 1.0
SEEDS = [67, 1, 2, 3, 4]

if RUN_TEST:
    fit_data = pd.concat([train, validation], ignore_index=True)

    summaries = [] 
    activity_tables = [] 
    time_tables = []

    for seed in SEEDS:
        model = HybridRecommender(factors=BEST_FACTORS,
                                  regularization=BEST_REGULARIZATION,
                                  alpha=BEST_ALPHA,
                                  random_state=seed,
                                  show_progress=False).fit(fit_data, movies)

        summary, per_user = evaluate(model, test)
        summaries.append({"seed": seed, **summary})
        activity_tables.append(by_user_activity(per_user, fit_data)) # cold-start / low / high
        time_tables.append(by_time_period(model, test)) # early / late half
        print(f"seed {seed}: NDCG@10 = {summary['NDCG@10']:.4f}", flush=True)

    # overall: mean and 95% confidence interval over seeds
    results = pd.DataFrame(summaries)
    n_seeds = len(results)
    t_value = stats.t.ppf(0.975, df=n_seeds - 1)

    print(f"\nTest results over {n_seeds} seeds (evaluated users: {int(results['Evaluated users'].iloc[0])})")
    for metric in ["Precision@10", "Recall@10", "NDCG@10", "Coverage"]:
        mean = results[metric].mean()
        half_width = t_value * results[metric].std(ddof=1) / np.sqrt(n_seeds)
        print(f"{metric:<13} {mean:.4f}  95% CI [{mean - half_width:.4f}, {mean + half_width:.4f}]")

    # breakdowns on test, averaged over seeds
    activity = pd.concat(activity_tables).groupby(level=0).mean()
    print("\nBy user activity (test, mean over seeds):")
    print(activity.round(4).to_string())

    numeric = ["Precision@10", "Recall@10", "NDCG@10", "Users"]
    time_mean = pd.concat(time_tables).groupby(level=0)[numeric].mean()
    time_mean.insert(0, "to", time_tables[0]["to"])
    time_mean.insert(0, "from", time_tables[0]["from"])
    print("\nBy time period (test, mean over seeds):")
    print(time_mean.loc[["early", "late"]].round(4).to_string())


FRACTIONS = [1.0, 0.5, 0.2]
SUBSAMPLE_SEED = 42

if RUN_SPARSITY:
    fit_data = pd.concat([train, validation], ignore_index=True)

    rows = []
    for fraction in FRACTIONS:
        sample = fit_data.sample(frac=fraction, random_state=SUBSAMPLE_SEED)

        seed_results = []
        for seed in SEEDS:
            model = HybridRecommender(factors=BEST_FACTORS,
                                      regularization=BEST_REGULARIZATION,
                                      alpha=BEST_ALPHA,
                                      random_state=seed,
                                      show_progress=False).fit(sample, movies)
            summary, per_user = evaluate(model, test)
            known = per_user["UserID"].isin(model.user_ids)
            summary["NDCG@10 known users"] = per_user.loc[known, "NDCG@10"].mean()
            summary["Cold-start users"] = int((~known).sum())
            seed_results.append(summary)
        seed_results = pd.DataFrame(seed_results)

        n_seeds = len(seed_results)
        t_value = stats.t.ppf(0.975, df=n_seeds - 1)
        ndcg_mean = seed_results["NDCG@10"].mean()
        ndcg_half = t_value * seed_results["NDCG@10"].std(ddof=1) / np.sqrt(n_seeds)

        rows.append({
            "Training data": f"{fraction:.0%}",
            "Ratings": len(sample),
            "Precision@10": seed_results["Precision@10"].mean(),
            "Recall@10": seed_results["Recall@10"].mean(),
            "NDCG@10": ndcg_mean,
            "NDCG@10 95% CI": f"[{ndcg_mean - ndcg_half:.4f}, {ndcg_mean + ndcg_half:.4f}]",
            "NDCG@10 known users": seed_results["NDCG@10 known users"].mean(),
            "Coverage": seed_results["Coverage"].mean(),
            "Evaluated users": int(seed_results["Evaluated users"].iloc[0]),
            "Cold-start users": int(seed_results["Cold-start users"].iloc[0]),
        })
        print(f"{fraction:.0%} of training data: NDCG@10 = {ndcg_mean:.4f}", flush=True)

    sparsity = pd.DataFrame(rows)
    print(f"\nSparsity ablation (test, mean over {len(SEEDS)} seeds, subsampling seed {SUBSAMPLE_SEED})")
    print(sparsity.round(4).to_string(index=False))