# DATA.ML.360-2026-Group7
Repository for DATA.ML.360 course project

## Setup

Start by pulling the source code from git. Move to project folder and create a virtual environment:
```bash
python -m venv .venv
python -m pip install numpy pandas scipy scikit-learn
```
Place `ml-1m.zip` beside `KNN_model.py`. The archive must contain `ml-1m/ratings.dat` and `ml-1m/movies.dat`; extraction is unnecessary.

Activate venv with:

Windows
```bash
.venv\scripts\activate
```
Linux or macOS
```bash
source .venv\bin\activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Download MovieLens 1M dataset and unzip into ml-1m folder. <https://grouplens.org/datasets/movielens/1m/>

## Running the project

After setting up code, venv, dependencies and dataset the project can be run using the following command
- ml-1m can be replaced with another data path if needed
- Plot will be saved to /results folder and model results are visible in terminal

```bash
python run_all.py --data-dir ml-1m
```

