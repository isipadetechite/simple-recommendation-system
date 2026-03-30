import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Create user-movie matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute similarity
similarity = cosine_similarity(user_movie_matrix.T)

# Convert to DataFrame for easy lookup
similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)


def recommend(movie_name):
    if movie_name not in similarity_df:
        print("Movie not found.")
        return

    similar_scores = similarity_df[movie_name].sort_values(ascending=False)

    print(f"\nTop recommendations for '{movie_name}':\n")
    print(similar_scores.iloc[1:6])


# Test
recommend("Toy Story (1995)")