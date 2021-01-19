
def create_users_book_matrix(ratings_df):
    """
    Return dataframe contains users as rows, books as columns and ratings as values

    :param ratings_df: ratings dataframe (user_id, book_id, rating)
    :return: dataframe contains users as rows, books as columns and ratings as values
    """
    users_books_ratings_matrix = ratings_df.pivot(index='user_id', columns='book_id', values='rating')
    return users_books_ratings_matrix