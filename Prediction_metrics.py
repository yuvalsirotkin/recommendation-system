import data_loaders
import collaborative_filtering
import numpy as np
import pandas as pd
import utils

def filter_high_scores(tests_df, minimal_high_score = 4):
    """

    :param tests_df: data to filter
    :param minimal_high_score: minimal score to filter (default - 4)
    :return: Ratings rated high
    """
    return tests_df.drop(tests_df[tests_df.rating < minimal_high_score].index)

def filter_users_less_than_k(tests_df, k):
    """

    :param tests_df: data to filter
    :param k: minimal number of book each user has to rate
    :return: Ratings rated by a user who rated more than k books
    """
    gb = tests_df.groupby('user_id')
    more_than_k = gb.filter(lambda x: len(x) >= k)
    return more_than_k

def filter_more_than_k_high(tests_df, k, minimal_high_score = 4):
    """

    :param tests_df: data to filter
    :param k: minimal number of book each user has to rate
    :param minimal_high_score: minimal score to filter (default - 4)
    :return: Ratings rated high by a user who rated more than k books
    """
    tests_df = filter_high_scores(tests_df, minimal_high_score)
    tests_df = filter_users_less_than_k(tests_df, k)
    return tests_df

def precision_k(k):
    """
    Calculate the proportion of relevant recommendation (books that the user rated with 4/5) in the books that the system recmmended of

    :param k: top k to check
    :return: the proportion
    """
    tests_df = data_loaders.load_tests()
    books_df = data_loaders.load_books()
    tests_df = filter_more_than_k_high(tests_df, k, minimal_high_score = 4)
    # test_df contains only the ratings of users who rated high (4/5) more than k books
    groups_by = tests_df.groupby('user_id')
    users_filtered = groups_by.groups.keys()
    sum_precision = 0
    for user in users_filtered:
        ids, hits = get_hits(groups_by, user, k, books_df)
        number_of_hits = len(hits)
        sum_precision = sum_precision + (number_of_hits / k)
    precision = sum_precision / len(users_filtered)
    return precision

def get_hits(groups_by, user, k, books_df):
    user_group = groups_by.get_group(user)
    books_titles = collaborative_filtering.get_CF_recommendation(user, k)
    # idx = books_titles.index
    ids = [books_df[books_df['title'] == book]['book_id'].values[0] for book in books_titles]
    hits = user_group[user_group['book_id'].isin(ids)]
    return ids, hits

def get_position_of_hits(hits, ids):
    hits_val = hits['book_id'].values
    positions = np.array([np.where(ids == hit)[0] for hit in hits_val]) + 1
    return positions

def ARHR(k):
    """
    Calculate the proportion of relevant recommendation (books that the user rated with 4/5) in the books that the system recmmended of

    :param k: top k to check
    :return: the proportion
    """
    tests_df = data_loaders.load_tests()
    books_df = data_loaders.load_books()
    tests_df = filter_more_than_k_high(tests_df, k, minimal_high_score = 4)
    # test_df contains only the ratings of users who rated high (4/5) more than k books
    groups_by = tests_df.groupby('user_id')
    users_filtered = groups_by.groups.keys()
    sum_precision = 0
    for user in users_filtered:
        ids , hits = get_hits(groups_by, user, k, books_df)
        number_of_hits = len(hits)
        if number_of_hits > 0:
            print("hit")
            positions = get_position_of_hits(hits, ids)
            sum_precision = sum_precision + sum(1 / positions)
    precision = sum_precision / len(users_filtered)
    return precision


def RMSE():
    """
    Calculate the proportion of relevant recommendation (books that the user rated with 4/5) in the books that the system recmmended of

    :param k: top k to check
    :return: the proportion
    """
    tests_df = data_loaders.load_tests()
    user_book_id_test_df = utils.create_users_book_matrix(tests_df)
    books_df = data_loaders.load_books()
    pred_np =  collaborative_filtering.build_CF_prediction_matrix('cosine')
    pred_pd = pd.DataFrame(pred_np)
    pred_pd = pred_pd.T.rename(books_df['book_id']).T
    pred_pd.index = np.arange(1, 5001)

    diff = pred_pd - user_book_id_test_df
    sum_of_powers = (diff ** 2).sum().sum()
    n = (~pd.isna(diff)).sum().sum()
    RMSE = (sum_of_powers / n) ** (0.5)
    return RMSE