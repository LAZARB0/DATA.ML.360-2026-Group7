# %% Import libraries
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from zipfile import ZipFile
import argparse
from load_data import load_data


# %% Load MovieLens 1M

def load_movielens(data_dir):
    ratings, _users, movies = load_data(data_dir)
    ratings = ratings.rename(columns={
        'UserID': 'user_id', 'MovieID': 'item_id',
        'Rating': 'rating', 'Timestamp': 'timestamp',
    })
    movies = movies.rename(columns={
        'MovieID': 'item_id', 'Title': 'title', 'Genres': 'genres',
    })
    return ratings, movies

# %% Mean-centered user-based kNN
class UserBasedKNN:
    """Predict from the mean-centered ratings of similar users."""

    def __init__(self, k=200, similarity='cosine', positive_threshold=4, min_common_items=2):
        self.k = k
        self.similarity = similarity
        self.positive_threshold = positive_threshold
        self.min_common_items = min_common_items
        self.fitted = False

    def fit(self, ratings, movies, seen_ratings=None):

        self.fitted = False
        # seen_ratings used to filter out already rated items for each user during recommendation. 
        # If not provided, use the training ratings.
        if seen_ratings is None:
            seen_ratings = ratings

        # Create a dictionary mapping user_id to a set of item_ids they have rated
        self.seen = seen_ratings.groupby('user_id').item_id.apply(set).to_dict()

        # Create a mapping from item_id to index and user_id to index for matrix construction
        self.movies=movies.set_index('item_id').sort_index().copy()
        self.item_ids=self.movies.index.to_numpy()
        self.user_ids=np.sort(ratings.user_id.unique())

        # Create a mapping from user_id to index for matrix construction
        self.user_to_index={uid:i for i,uid in enumerate(self.user_ids)}

        shape=(len(self.user_ids),len(self.item_ids))

        # Create sparse matrices for observed ratings and rating values
        ui=np.searchsorted(self.user_ids,ratings.user_id)
        ii=np.searchsorted(self.item_ids,ratings.item_id)

        self.observed=csr_matrix((np.ones(len(ratings)),(ui,ii)),shape=shape)
        self.rating_matrix=csr_matrix((ratings.rating.to_numpy(dtype=float),(ui,ii)),shape=shape)

        # Compute user means and global mean for mean-centering
        counts=np.asarray(self.observed.sum(axis=1)).ravel()
        sums=np.asarray(self.rating_matrix.sum(axis=1)).ravel()
        self.global_mean=float(ratings.rating.mean()) if len(ratings) else 3.0
        self.user_means=np.divide(sums,counts,out=np.full(len(counts),self.global_mean),where=counts>0)
        deviations=ratings.rating.to_numpy(dtype=float)-self.user_means[ui]

        # Create a sparse matrix for mean-centered ratings
        self.centered_ratings=csr_matrix((deviations,(ui,ii)),shape=shape)
        self.centered_ratings.eliminate_zeros()
        self.similarities = self._user_similarities()
        return self.set_k(self.k)

    def set_k(self, k):
        """Select positive neighbors directly"""
        if k < 1:
            raise ValueError('k must be positive')
        self.k = k
        ids = self.user_ids
        rows, cols, weights = [], [], []
        for target, similarities in enumerate(self.similarities):
            candidates = np.flatnonzero(similarities > 1e-12)
            candidates = candidates[candidates != target]
            order = np.lexsort((ids[candidates], -similarities[candidates]))[:k]
            selected = candidates[order]
            rows.extend([target] * len(selected))
            cols.extend(selected.tolist())
            weights.extend(similarities[selected].tolist())
        self.weights = csr_matrix((weights, (rows, cols)), shape=(len(ids), len(ids)))
        self.absolute_weights = abs(self.weights)
        self.fitted = True
        return self

    def _user_similarities(self):
        """Compute user similarities over co-rated items."""
        observed = self.observed
        centered = self.centered_ratings
        raw = self.rating_matrix
        common=(observed@observed.T).toarray()
        if self.similarity=='cosine':
            # Use mean-centered ratings for cosine similarity
            x=centered
            numerator=(x@x.T).toarray()
            squares=(x.multiply(x)@observed.T).toarray()
            denominator=np.sqrt(np.maximum(squares*squares.T,0.0))
        else:
            # Use raw ratings for Pearson correlation
            x=raw
            pair_sums=(x@observed.T).toarray()
            numerator=(x@x.T).toarray()
            numerator-=np.divide(pair_sums*pair_sums.T,common,
                                 out=np.zeros_like(common),where=common>0)
            squares=(x.multiply(x)@observed.T).toarray()
            squares-=np.divide(pair_sums**2,common,out=np.zeros_like(common),where=common>0)
            np.maximum(squares,0.0,out=squares)
            denominator=np.sqrt(squares*squares.T)
        valid=(common>=self.min_common_items)&(denominator>1e-12)
        similarity=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=valid)
        np.clip(similarity,-1.0,1.0,out=similarity)
        np.fill_diagonal(similarity,0.0)
        return similarity

    def predict_scores(self,user_id):
        """Return the predicted scores for a given user."""
        u=self.user_to_index.get(user_id)
        baseline=(
            self.global_mean if u is None
            else self.user_means[u]
        )
        scores=np.full(len(self.item_ids),baseline,dtype=float)
        denominator=np.zeros(len(self.item_ids),dtype=float)
        if u is not None:
            numerator=(self.weights.getrow(u)@self.centered_ratings).toarray().ravel()
            denominator=(self.absolute_weights.getrow(u)@self.observed).toarray().ravel()
            scores+=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12)
        return scores,denominator

    def recommend(self,user_id,n=10,full_catalogue=False):
        """Return the top-n recommended items for a given user, excluding already rated items."""
        scores,denominator=self.predict_scores(user_id)
        # Filter out items that the user has already rated.
        seen_items = self.seen.get(user_id, set())
        scores[np.isin(self.item_ids, list(seen_items))] = -np.inf
        valid=np.isfinite(scores)
        if not full_catalogue:
            valid&=denominator>1e-12
        candidates=np.flatnonzero(valid)

        # Sort the candidates by score and return the top-n recommended items along with their scores and neighbor weight sums.
        order=np.lexsort((self.item_ids[candidates],-scores[candidates]))
        selected=candidates[order[:n]]
        result=self.movies.loc[self.item_ids[selected]].reset_index()
        result['score']=scores[selected]
        result['neighbor_weight_sum']=denominator[selected]
        return result[['item_id','title','genres','score','neighbor_weight_sum']]

# %% Temporal split

def temporal_split(ratings):
    # The ratings are split into three parts based on timestamps: 80% for training, 10% for validation, and 10% for testing.
    timestamps = np.sort(ratings.timestamp.to_numpy())
    cut1 = timestamps[int(len(timestamps) * 0.8)]
    cut2 = timestamps[int(len(timestamps) * 0.9)]
    train = ratings[ratings.timestamp < cut1].copy()
    validation = ratings[
        (ratings.timestamp >= cut1) & (ratings.timestamp < cut2)
    ].copy()
    test = ratings[ratings.timestamp >= cut2].copy()
    return train, validation, test


# %% Metrics for one user
def ranking_metrics(recommended_items, relevant_items, at=10):
    relevant = set(relevant_items)
    recommended = list(recommended_items)[:at]
    hits = np.array([item in relevant for item in recommended], dtype=float)
    dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
    idcg = np.sum(1 / np.log2(np.arange(2, min(at, len(relevant)) + 2)))
    return {
        f'precision_at_{at}': float(hits.sum() / at),
        f'recall_at_{at}': float(hits.sum() / len(relevant)),
        f'ndcg_at_{at}': float(dcg / idcg),
    }



def nested_sample(frame, fraction, seed):
    """Same permutation across fractions; no heldout sampling."""
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    order = np.random.default_rng(seed).permutation(len(frame))
    return frame.iloc[order[:max(1, int(len(frame) * fraction))]].copy()


def evaluate(model, heldout, activity=None, activity_cut=None, time_cut=None, at=10):
    """Full-catalogue ranking; temporal slices reuse the same recommendation list."""
    positive = heldout[heldout.rating >= model.positive_threshold]
    rows, catalogue = [], set()
    for uid, events in positive.groupby('user_id'):
        seen = model.seen.get(uid, set())
        relevant = set(events.item_id) - seen
        if not relevant:
            continue
        recommended = model.recommend(uid, n=at, full_catalogue=True).item_id.tolist()
        catalogue.update(recommended)
        groups = [('overall', 'all', relevant)]
        if activity is not None:
            label = 'low' if activity.get(uid, 0) <= activity_cut else 'high'
            groups.append(('user_activity', label, relevant))
        if time_cut is not None:
            for label, mask in [('early', events.timestamp < time_cut),
                                ('late', events.timestamp >= time_cut)]:
                subset = set(events.loc[mask, 'item_id']) - seen
                if subset:
                    groups.append(('test_period', label, subset))
        for dimension, group, truth in groups:
            rows.append(dict(user_id=uid, dimension=dimension, group=group,
                             relevant_items=len(truth),
                             **ranking_metrics(recommended, truth, at)))
    columns = ['user_id', 'dimension', 'group', 'relevant_items',
               f'precision_at_{at}', f'recall_at_{at}', f'ndcg_at_{at}']
    return pd.DataFrame(rows, columns=columns), len(catalogue) / len(model.item_ids)


def bootstrap_interval(values, repeats, seed):
    """Percentile interval for the user-macro mean, resampling users."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for start in range(0, repeats, 100):
        size = min(100, repeats-start)
        means[start:start+size] = values[rng.integers(len(values), size=(size, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))

# %%
def run_experiment(data_path, k_values=(10, 20, 50, 100, 200, 400),
                   fractions=(1.0, 0.5, 0.2), seed=66, bootstrap=2000):
    import time
    ratings, movies = load_movielens(data_path)
    train, validation, test = temporal_split(ratings)
    fit_data = pd.concat([train, validation], ignore_index=True)
    # Fixed cohort membership for comparable results across sparsity levels.
    activity = train.groupby('user_id').size().to_dict()
    activity_cut = float(np.median(list(activity.values())))
    time_cut = (int(test.timestamp.min()) + int(test.timestamp.max())) / 2
    similarities = ('cosine', 'pearson')
    metrics = ('precision_at_10', 'recall_at_10', 'ndcg_at_10')
    summaries = []
    for fraction in fractions:
        sampled_train = nested_sample(train, fraction, seed)
        sampled_final = nested_sample(fit_data, fraction, seed)
        candidates = []
        for similarity in similarities:
            start = time.perf_counter()
            model = UserBasedKNN(k_values[0], similarity).fit(sampled_train, movies, seen_ratings=train)
            for k in k_values:
                model.set_k(k)
                per_user, _ = evaluate(model, validation)
                row = dict(fraction=fraction, similarity=similarity, k=k,
                           **per_user[list(metrics)].mean().to_dict())
                candidates.append(row)
                print('Validation:', row, flush=True)
            del model
        best = sorted(candidates, key=lambda r: (-r['ndcg_at_10'], r['k'], r['similarity']))[0]
        print('Selected:', best, flush=True)
        model = UserBasedKNN(best['k'], best['similarity']).fit(sampled_final, movies, seen_ratings=fit_data)
        per_user, coverage = evaluate(model, test, activity, activity_cut, time_cut)
        per_user['model'], per_user['fraction'] = 'user', fraction
        for (dimension, group), frame in per_user.groupby(['dimension', 'group']):
            row = dict(fraction=fraction, model='user', dimension=dimension, group=group,
                       users=len(frame), relevant_items=int(frame.relevant_items.sum()),
                       training_rows=len(sampled_final), training_users=sampled_final.user_id.nunique(),
                       training_items=sampled_final.item_id.nunique())
            if dimension == 'overall':
                row['catalogue_coverage'] = coverage
                row['excluded_users'] = test.user_id.nunique()-len(frame)
            for metric in metrics:
                row[metric] = frame[metric].mean()
                lo, hi = bootstrap_interval(frame[metric], bootstrap, seed)
                row[metric+'_ci_low'], row[metric+'_ci_high'] = lo, hi
            summaries.append(row)
        del model
    summary_results = pd.DataFrame(summaries)
    overall = summary_results.query("dimension == 'overall'").copy()
    print('Overall and subgroup results with 95% confidence intervals:', flush=True)
    print(summary_results.to_string(index=False), flush=True)
    print('Sparsity results:', flush=True)
    print(overall[['fraction', 'ndcg_at_10']].to_string(index=False), flush=True)
    return overall


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    parser.add_argument("--data", "--data-dir",dest="data", type=Path, default=base/'ml-1m.zip')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--bootstrap', type=int, default=2000)
    # Notebook cells inherit kernel arguments such as --f=<connection file>.
    # Explicit argv and normal script/%run arguments still use strict parsing.
    if argv is None and Path(sys.argv[0]).stem == 'ipykernel_launcher':
        argv = []
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error('Use at least 100 bootstrap replicates')
    return run_experiment(args.data, seed=args.seed,
                   bootstrap=args.bootstrap)


if __name__ == '__main__':
    main()

# %%
