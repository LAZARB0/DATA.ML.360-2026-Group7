#%% import libraries
import argparse
import ast
from collections import defaultdict
import contextlib
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys
import time
from zipfile import ZipFile


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


#%% set seed, hyperparameters and models
SEED = 50
K = 10
BOOTSTRAP = 2000
METRICS = ['precision_at_10', 'recall_at_10', 'ndcg_at_10']
FILES = {'UserKNN': 'KNN_user_based_model.py', 'ItemKNN': 'KNN_item_based_model.py',
         'MostPopular': 'most_popular.py', 'Markov': 'markov_model.py', 'Hybrid': 'hybrid.py'}
RENAME = {'UserID': 'user_id', 'MovieID': 'item_id', 'Rating': 'rating',
          'Timestamp': 'timestamp', 'Title': 'title', 'Genres': 'genres'}

# %% 
def definitions(path, names, extra=None):
    # Original files have top-level experiments. Load only unchanged definitions.
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing definitions: {path}')
    env = dict(np=np, pd=pd, csr_matrix=csr_matrix, NearestNeighbors=NearestNeighbors,
               defaultdict=defaultdict, math=math, log2=math.log2, K=K, POSITIVE_THRESHOLD=4)
    env.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def load_models(folder):
    from implicit.bpr import BayesianPersonalizedRanking
    return {
        'UserKNN': definitions(folder / FILES['UserKNN'], ['UserBasedKNN']),
        'ItemKNN': definitions(folder / FILES['ItemKNN'], ['ItemBasedKNN']),
        'MostPopular': definitions(folder / FILES['MostPopular'], ['most_popular_rankings', 'recommend']),
        'Markov': definitions(folder / FILES['Markov'], ['build_markov_model', 'recommend']),
        'Hybrid': definitions(folder / FILES['Hybrid'], ['HybridRecommender'],
                              {'BayesianPersonalizedRanking': BayesianPersonalizedRanking})}


def load_data(path):
    def read(r, m):
        return (pd.read_csv(r, sep='::', engine='python', names=['UserID','MovieID','Rating','Timestamp']),
                pd.read_csv(m, sep='::', engine='python', encoding='latin1', names=['MovieID','Title','Genres']))
    if path.is_dir():
        return read(path/'ratings.dat', path/'movies.dat')
    with ZipFile(path) as z:
        return read(z.open('ml-1m/ratings.dat'), z.open('ml-1m/movies.dat'))


def split_data(ratings):
    t = np.sort(ratings.Timestamp.to_numpy())
    a, b = t[int(.8*len(t))], t[int(.9*len(t))]
    return (ratings[ratings.Timestamp < a].copy(),
            ratings[(ratings.Timestamp >= a) & (ratings.Timestamp < b)].copy(),
            ratings[ratings.Timestamp >= b].copy())


def seen_items(data):
    return data.groupby('UserID').MovieID.apply(set).to_dict()


def relevant_items(heldout, seen):
    groups = heldout[heldout.Rating >= 4].groupby('UserID').MovieID.apply(set)
    relevant = {uid: items-seen.get(uid, set()) for uid, items in groups.items()}
    relevant = {uid: items for uid, items in relevant.items() if items}
    return relevant

# %%
class Model:
    def __init__(self, name, env, data, movies, config):
        self.name, self.env = name, env
        self.seen = seen_items(data)
        self.catalogue = set(movies.MovieID)
        if name in ('UserKNN', 'ItemKNN'):
            cls = env['UserBasedKNN' if name == 'UserKNN' else 'ItemBasedKNN']
            self.model = cls(**config).fit(data.rename(columns=RENAME), movies.rename(columns=RENAME))
        elif name == 'Hybrid':
            self.model = env['HybridRecommender'](**config, random_state=SEED,
                iterations=100, num_threads=1, show_progress=False).fit(data, movies)
        elif name == 'MostPopular':
            with contextlib.redirect_stdout(io.StringIO()):
                self.rankings = env['most_popular_rankings'](data)
            self.ranking = self.rankings[config['variant']]
        else:
            env['transition_probs'] = env['build_markov_model'](data, min_count=config['min_count'])
            env['popular_items'] = data.MovieID.value_counts().index.tolist()
            self.last = data.sort_values(['UserID','Timestamp']).groupby('UserID').MovieID.last().to_dict()

    def recommend(self, uid):
        if self.name in ('UserKNN', 'ItemKNN'):
            recs = self.model.recommend(uid, n=K, full_catalogue=True).item_id.tolist()
        elif self.name == 'Hybrid':
            recs = self.model.recommend(uid, n=K)
        elif self.name == 'MostPopular':
            recs = self.env['recommend'](self.ranking, self.seen.get(uid,set()), k=K)
        else:
            # Current source requires seen_items. Cold users use its popularity fallback.
            recs = self.env['recommend'](self.last.get(uid), self.seen.get(uid,set()), k=K)
        # Keep native rankings: do not append arbitrary items to short lists.
        result = []
        for item in recs:
            if item in self.catalogue and item not in self.seen.get(uid,set()) and item not in result:
                result.append(int(item))
        return result[:K]


def evaluate(model, relevant):
    discounts = 1/np.log2(np.arange(2,K+2))
    rows = []
    for uid in sorted(relevant):
        truth = relevant[uid]
        recs = model.recommend(uid)
        hits = np.array([i in truth for i in recs], dtype=float)
        rows.append(dict(user_id=uid, precision_at_10=hits.sum()/K,
                         recall_at_10=hits.sum()/len(truth),
                         ndcg_at_10=(hits @ discounts[:len(hits)])/discounts[:min(K,len(truth))].sum()))
    return pd.DataFrame(rows).set_index('user_id')


def grid(name):
    if name in ('UserKNN','ItemKNN'):
        return [dict(k=k,similarity=s) for s in ('cosine','pearson') for k in (10,20,50,100,200,400)]
    if name == 'Hybrid':
        return [dict(factors=f,regularization=r,alpha=a) for f,r,a in itertools.product((50,100),(.01,.1),(.75,.9,1.))]
    if name == 'Markov':
        return [dict(min_count=n) for n in (1,3,5)]  # original grid, with recommendation k fixed at 10
    return [dict(variant=v) for v in ('count (all ratings)','positive (rating >= 4)','average (min count = mean)')]


def tune(name, env, train, movies, relevant):
    best, best_score, model, previous = None, -np.inf, None, None
    print(f'\nValidation training: {name} (seed={SEED})', flush=True)
    for config in grid(name):
        if name == 'UserKNN':
            key = config['similarity']
        elif name == 'Hybrid':
            key = (config['factors'],config['regularization'])
        elif name == 'MostPopular':
            key = 'all'
        else:
            key = json.dumps(config)
        if model is not None and key == previous:
            if name == 'UserKNN':
                model.model.set_k(config['k'])
            elif name == 'Hybrid':
                model.model.alpha = config['alpha']
            else:
                model.ranking = model.rankings[config['variant']]
        else:
            model = None
            #print(f'Fitting {name}: {config}', flush=True)
            start = time.perf_counter()
            model = Model(name,env,train,movies,config)
            #print(f'Fit completed in {time.perf_counter()-start:.1f}s', flush=True)
            previous = key
        scores = evaluate(model,relevant).mean()
        #print(f'Validation {config}: ' + ', '.join(f'{m}={scores[m]:.6f}' for m in METRICS), flush=True)
        # First in predeclared grid breaks ties; never select using test results.
        if scores.ndcg_at_10 > best_score:
            best, best_score = config.copy(), scores.ndcg_at_10
    print(f'Selected {name}: {best}', flush=True)
    return best


def mean_ci(values):
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAP)
    for start in range(0,BOOTSTRAP,100):
        size = min(100,BOOTSTRAP-start)
        means[start:start+size] = values[rng.integers(len(values),size=(size,len(values)))].mean(axis=1)
    return np.quantile(means,[.025,.975])


def pairwise(per_user):
    rows = []
    for a,b in itertools.combinations(per_user,2):
        x,y = per_user[a].sort_index(),per_user[b].sort_index()
        for metric in METRICS:
            delta = x[metric].to_numpy()-y[metric].to_numpy()
            lo,hi = mean_ci(delta)
            rows.append(dict(model_a=a,model_b=b,metric=metric,
                             difference_a_minus_b=delta.mean(),ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows)


def compare(folder, data_path):
    folder = Path(folder)
    hashes = lambda: {name:hashlib.sha256((folder/file).read_bytes()).hexdigest() for name,file in FILES.items()}
    before = hashes()
    models = load_models(folder)
    ratings,movies = load_data(Path(data_path))
    train,validation,test = split_data(ratings)
    fit = pd.concat([train,validation],ignore_index=True)
    validation_users = relevant_items(validation,seen_items(train))
    test_users = relevant_items(test,seen_items(fit))
    selected = {name:tune(name,models[name],train,movies,validation_users) for name in FILES}
    per_user,rows = {},[]
    for name in FILES:
        print(f'\nFinal training: {name}, {selected[name]}, seed={SEED}', flush=True)
        start = time.perf_counter()
        model = Model(name,models[name],fit,movies,selected[name])
        per_user[name] = evaluate(model,test_users)
        for metric in METRICS:
            lo,hi = mean_ci(per_user[name][metric])
            rows.append(dict(model=name,metric=metric,mean=per_user[name][metric].mean(),ci_low=lo,ci_high=hi))
        del model
        print(f'Training and evaluation completed in {time.perf_counter()-start:.1f}s', flush=True)
    summary = pd.DataFrame(rows)
    differences = pairwise(per_user)
    if hashes() != before:
        raise RuntimeError('Original model files changed during execution')
    print('\nIndividual model results and 95% CI:', flush=True)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'), flush=True)
    print('\nModel differences (A minus B) and 95% CI:', flush=True)
    print(differences.to_string(index=False,float_format=lambda x:f'{x:.6f}'), flush=True)
    return summary,differences


def main(argv=None):
    base = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=base)
    parser.add_argument('--data',type=Path)
    if argv is None and Path(sys.argv[0]).stem == 'ipykernel_launcher':
        argv = []
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        return compare(args.model_dir,args.data or args.model_dir/'ml-1m.zip')


if __name__ == '__main__':
    results = main()
