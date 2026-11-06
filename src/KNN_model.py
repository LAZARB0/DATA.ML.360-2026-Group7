
# %% Import libraries
import argparse

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

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


# %% Mean-centered Item-based kNN
class ItemBasedKNN:
    """ kNN: pred(u,i)=avg(u)+sum(sim(u,v)*(r(v,i)-avg(v)))/sum(abs(sim(u,v))) over the k neighbors v
    """

    def __init__(self, k=200, similarity='cosine', positive_threshold=4, min_common_users=2):
        self.k=k
        self.similarity=similarity
        self.positive_threshold=positive_threshold 
        self.min_common_users=min_common_users
        self.fitted=False

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
        similarities=self._item_similarities()

        distances=1.0-similarities
        np.fill_diagonal(distances,0.0)
        self.neighbor_model=NearestNeighbors(metric='precomputed',algorithm='brute',
                                             n_neighbors=min(self.k+1,len(self.item_ids)),n_jobs=1)
        self.neighbor_model.fit(distances)
        sources, targets, weights = [], [], []
        for start in range(0,len(self.item_ids),128):
            batch_distances,batch_indices=self.neighbor_model.kneighbors(
                distances[start:start+128],n_neighbors=len(self.item_ids))
            for offset,(distance,index) in enumerate(zip(batch_distances,batch_indices)):
                target=start+offset
                sim=1.0-distance
                keep=(index!=target)&(sim>1e-12)
                index,sim=index[keep],sim[keep]
                order=np.lexsort((self.item_ids[index],-sim))[:self.k]

                # Append the selected neighbors and their similarities to the lists
                sources.extend(index[order].tolist())
                targets.extend([target] * len(order))
                weights.extend(sim[order].tolist())
        
        # Create a sparse matrix for item weights based on the selected neighbors and their similarities
        self.item_weights = csr_matrix(
            (weights, (sources, targets)),
            shape=(len(self.item_ids), len(self.item_ids)),
        )
        self.absolute_weights = abs(self.item_weights)
        self.fitted = True
        return self

    def _item_similarities(self):
        """Compute item-item similarities based on mean-centered ratings."""
        common=(self.observed.T@self.observed).toarray()
        if self.similarity=='cosine':
            # Use mean-centered ratings for cosine similarity
            x=self.centered_ratings
            numerator=(x.T@x).toarray()
            squares=(x.multiply(x).T@self.observed).toarray()
            denominator=np.sqrt(np.maximum(squares*squares.T,0.0))
        else:
            # Use raw ratings for Pearson correlation
            x=self.rating_matrix
            pair_sums=(x.T@self.observed).toarray()
            numerator=(x.T@x).toarray()
            numerator-=np.divide(pair_sums*pair_sums.T,common,
                                 out=np.zeros_like(common),where=common>0)
            squares=(x.multiply(x).T@self.observed).toarray()
            squares-=np.divide(pair_sums**2,common,out=np.zeros_like(common),where=common>0)
            np.maximum(squares,0.0,out=squares)
            denominator=np.sqrt(squares*squares.T)
        valid=(common>=self.min_common_users)&(denominator>1e-12)
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
            numerator=(self.centered_ratings.getrow(u)@self.item_weights).toarray().ravel()
            # The denominator is the sum of absolute similarities for the items that the user has rated. This is used to normalize the predicted scores.
            denominator=(self.observed.getrow(u)@self.absolute_weights).toarray().ravel()
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


# %% Full-catalogue evaluation
def evaluate_model(model, heldout, at=10):
    relevant_by_user = (
        heldout[heldout.rating >= model.positive_threshold]
        .groupby('user_id').item_id.apply(set)
    )
    rows = []
    recommended_catalogue = set()
    for user_id, relevant in relevant_by_user.items():
        relevant = relevant - model.seen.get(user_id, set())
        if not relevant:
            continue
        recommendations = model.recommend(user_id, n=at, full_catalogue=True)
        recommended_items = recommendations.item_id.tolist()
        recommended_catalogue.update(recommended_items)
        rows.append({
            'user_id': user_id,
            **ranking_metrics(recommended_items, relevant, at),
        })
    per_user = pd.DataFrame(rows)
    columns = [f'precision_at_{at}', f'recall_at_{at}', f'ndcg_at_{at}']
    summary = per_user[columns].mean().to_dict()
    summary['catalogue_coverage'] = len(recommended_catalogue) / len(model.item_ids)
    summary['evaluated_users'] = len(per_user)
    summary['excluded_users'] = int(heldout.user_id.nunique() - len(per_user))
    return summary, per_user


# %% Four training filter modes
def filter_training(ratings, mode='none', min_user_ratings=20, min_item_ratings=20):
    """Filter the training data based on the specified mode.
    Apply both conditions to items and users simultaneously.
    """
    if mode == 'none':
        return ratings.copy()
    if mode == 'users':
        counts = ratings.groupby('user_id').size()
        return ratings[ratings.user_id.map(counts) >= min_user_ratings].copy()
    if mode == 'items':
        counts = ratings.groupby('item_id').size()
        return ratings[ratings.item_id.map(counts) >= min_item_ratings].copy()
    if mode == 'items&users':
        user_counts = ratings.groupby('user_id').size()
        item_counts = ratings.groupby('item_id').size()
        keep = (
            (ratings.user_id.map(user_counts) >= min_user_ratings)
            & (ratings.item_id.map(item_counts) >= min_item_ratings)
        )
        return ratings[keep].copy()
    raise ValueError("mode must be 'none', 'users','items&users' or 'items'.")


# %% Run and print results
def main(data_path=None, k_values=(10, 20, 50, 100, 200, 400),
         similarities=('cosine', 'pearson'), min_user_ratings=20,
         min_item_ratings=20, positive_threshold=4, min_common_users=2):
    if data_path is None:
        data_path = 'ml-1m'
    ratings, movies = load_movielens(data_path)
    train, validation, test = temporal_split(ratings)
    print('train / validation / test rows:', len(train), len(validation), len(test), flush=True)

    # Fit and evaluate models on the validation set to select the best hyperparameters (k and similarity).
    validation_rows = []
    for similarity in similarities:
        for k in k_values:
            print('Fitting:', similarity, 'k =', k, flush=True)
            model = ItemBasedKNN(k, similarity, positive_threshold, min_common_users)
            model.fit(train, movies)
            summary, _ = evaluate_model(model, validation)
            row = {'split': 'validation', 'similarity': similarity, 'k': k, **summary}
            validation_rows.append(row)
            print(row, flush=True)

    validation_results = pd.DataFrame(validation_rows)
    best = validation_results.sort_values(
        ['ndcg_at_10', 'k', 'similarity'], ascending=[False, True, True]
    ).iloc[0]
    best_k, best_similarity = int(best['k']), best['similarity']
    print('Selected:', best_similarity, 'k =', best_k, flush=True)

    # Fit the model on the combined training and validation data
    # evaluate it on the test set using different filtering modes.
    fit_data = pd.concat([train, validation], ignore_index=True)
    test_rows = []
    for mode in ['none', 'users', 'items', 'items&users']:
        filtered = filter_training(fit_data, mode, min_user_ratings, min_item_ratings)
        model = ItemBasedKNN(best_k, best_similarity, positive_threshold, min_common_users)
        model.fit(filtered, movies, seen_ratings=fit_data)
        summary, _ = evaluate_model(model, test)
        test_rows.append({
            'split': 'test', 'filter': mode,
            'similarity': best_similarity, 'k': best_k,
            'min_user_ratings': min_user_ratings if mode in ('users', 'items&users') else 0,
            'min_item_ratings': min_item_ratings if mode in ('items', 'items&users') else 0,
            'removed_rows': len(fit_data) - len(filtered),
            'training_users': filtered.user_id.nunique(),
            'training_items': filtered.item_id.nunique(),
            **summary,
        })

    print('Test results: none / users / items / items&users', flush=True)
    for row in test_rows:
        print(row, flush=True)
    return validation_results, pd.DataFrame(test_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='ml-1m')
    args = parser.parse_args()
    main(data_path=args.data_dir)
