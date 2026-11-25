#s
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking   # pip install implicit
from load_data import load_data
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

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
        self.pop_scores = (positive_train["MovieID"].value_counts().reindex(self.movie_ids, fill_value=0).to_numpy(dtype=float)) # popularity score array for cold-start users

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

    def candidates(self, user_id, n=100):
        if self._user_row(user_id) is None:
            scores = self.pop_scores.copy()
        else:
            scores = (self.alpha * self._minmax(self.bpr_scores(user_id)) + (1 - self.alpha) * self._minmax(self.content_scores(user_id)))

        seen = list(self.seen.get(user_id, set()))
        scores[np.searchsorted(self.movie_ids, seen)] = -np.inf
        top = np.flip(np.argsort(scores))[:n]
        return top, scores[top]

BEST_FACTORS = 100 
BEST_REGULARIZATION = 0.1
BEST_ALPHA = 1.0
SEED = 67

model = HybridRecommender(factors=BEST_FACTORS, regularization=BEST_REGULARIZATION,
                          alpha=BEST_ALPHA, random_state=SEED,
                          show_progress=False).fit(train, movies)

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

def genre_vectors(movies):
    indexed = movies.sort_values("MovieID").set_index("MovieID")
    genre_df = indexed["Genres"].str.get_dummies(sep="|")
    genres = genre_df.to_numpy(dtype=float)
    lengths = np.linalg.norm(genres, axis=1, keepdims=True) # sqrt(number of genres)
    unit = genres / lengths # each value in a row is divided by that row's length 
    return genres, unit

def mmr_rerank(positions, scores, unit, lam, k=10):
    # scale the scores
    minimum, maximum = np.min(scores), np.max(scores) 
    if maximum == minimum: 
        relevance = np.zeros(len(scores))
    else:
        relevance = (scores - minimum) / (maximum - minimum)

    similarity = unit[positions] @ unit[positions].T    
    selected = []
    max_sim = np.zeros(len(positions)) # nothing picked -> no redundancy
    for _ in range(k):
        mmr = lam * relevance - (1 - lam) * max_sim
        mmr[selected] = -np.inf # same pick can't come twice -> already picked similarity score=-inf
        best = np.argmax(mmr) # best maximising
        selected.append(best)
        max_sim = np.maximum(max_sim, similarity[best]) # update the redundancy     

    return positions[selected]

def intra_list_diversity(list_positions, unit):
    """measures how different recommended items are from one another with in a user's list
    average pairwise cosine distance"""
     
    similarity = unit[list_positions] @ unit[list_positions].T # similarity matrix    
    distance = 1 - similarity
    pairs = np.triu_indices(len(list_positions), k=1) # average over all PAIRS (not the diagonal, not counting each pair twice)

    return np.mean(distance[pairs])

def gini(exposure):
    """do a few movies get almost all the recommendations, or are recommendations
    spread across many movies"""
    sorted_values = np.sort(exposure) # smallest to largest
    n = len(sorted_values)
    # G = Σ (2i − n − 1) × xᵢ / (n × Σ xᵢ) for i = 1 … n
    total = 0
    for i in range(n):
        total += (2*(i+1)-n-1)*sorted_values[i]
    if np.sum(sorted_values) == 0:
        return 0
    G = total / (n*np.sum(sorted_values))
    return G

def genre_distribution(movie_positions, genres):
    # calibration error uses
    genre_rows = genres[movie_positions]
    n_genres = np.count_nonzero(genre_rows, axis=1, keepdims=True) # each row divided by its number of genres
    weighted = genre_rows / n_genres
    totals = weighted.sum(axis=0)

    return totals / totals.sum()

def calibration_error(p, list_positions, genres, a=0.01):
    """does the list's genre mix match the user's taste
    0=the list has exactly same genre, higher: the list over-represents some genres,
    misses others the user likes"""
    q = genre_distribution(list_positions, genres)
    q = (1-a)*q+a*p # q = (1 - a) * q + a * p
    mask = p > 0  # KL over genres where p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


