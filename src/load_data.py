import pandas as pd

# =================
# Load MovieLens 1M
# =================
def load_data(data_dir="ml-1m"):
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    users = pd.read_csv(
        f"{data_dir}/users.dat",
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
    )

    movies = pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["MovieID", "Title", "Genres"],
    )

    return ratings, users, movies