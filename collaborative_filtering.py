import data_loaders
import numpy as np
import utils
from sklearn.metrics.pairwise import pairwise_distances
import heapq
import pandas as pd


global_k = 0

ratings_df = data_loaders.load_ratings_pd()
books_data_df = data_loaders.load_books()
# create matrix - [i,j] cell contain the rating of user i to book j
users_books_ratings_matrix_pd = utils.create_users_book_matrix(ratings_df)
users_books_ratings_matrix_np = users_books_ratings_matrix_pd.to_numpy()


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0 # replace anything lower than the cut off with 0
    return arr

def build_CF_prediction_matrix(sim):
    """

    :param sim: similarity metric
    :return: collaborative filtering matrix
    """

    # normalize users_books_ratings_matrix_pd matrix
    global global_k
    k = global_k
    mean_user_rating = users_books_ratings_matrix_pd.mean(axis= 1).to_numpy().reshape(-1,1)
    ratings_diff = (users_books_ratings_matrix_np - mean_user_rating)
    ratings_diff[np.isnan(ratings_diff)] = 0
    # calc user similarity matrix ( how much 2 users similar to each other)
    user_similarity = 1 - pairwise_distances(ratings_diff, metric = sim)
    if (k != 0):
        user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    # calc prediction matrix according to user similarity
    # rows- users. cols- book. [i,j] contain the prediction for user i+1 about book j
    prediction_matrix = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return prediction_matrix


def get_CF_recommendation(user_id, k):
    """

    :param user_id: user to predict for
    :param k: number of books to return
    :return: top k books recommended for user
    """
    global global_k
    global_k = k
    sim = 'cosine'
    prediction_matrix= build_CF_prediction_matrix(sim)
    pred_pd = pd.DataFrame(prediction_matrix)
    pred_pd = pred_pd.T.rename(books_data_df['book_id']).T
    pred_pd.index = np.arange(1, 5001)

    # prediction for user
    pred_for_user = pred_pd.loc[user_id]
    # real ratings of the user
    users_books_ratings_matrix_pd_for_user = users_books_ratings_matrix_pd.loc[user_id]
    # check only unrated books because we don't want to recommend books that the user already read
    was_not_rated = np.isnan(users_books_ratings_matrix_pd_for_user)
    books_id = was_not_rated[was_not_rated].index.values
    predicted_ratings_unrated = pred_for_user[was_not_rated]
    # check which unrated books got the highest score
    idx = np.argsort(-predicted_ratings_unrated)
    id = books_id[idx[0:k]]
    # Return top k book
    return [books_data_df[books_data_df['book_id'] == user]['title'].values[0] for user in id]