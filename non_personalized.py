import data_loaders
import utils


def normalize_ratings(users_book_rating_matrix_df):
    """
    normalize ratings according to users

    :param users_book_rating_matrix_df: ratings df
    :return: normalized ratings df
    """
    mean_user_rating = users_book_rating_matrix_df.mean(axis=1).to_numpy().reshape(-1, 1)
    ratings_normal_df = (users_book_rating_matrix_df - mean_user_rating)
    return ratings_normal_df


def add_title_column(df_each_row_book, books_df):
    """
    add title column to the data frame

    :param users_book_rating_matrix_df: dataframe contain books as rows. The index of the rows is the book id
    :param books_df: books dataframe- contain books id and titles
    :return:
    """
    titles = books_df['title'].iloc[df_each_row_book.index - 1]
    titles.index = titles.index + 1
    df_each_row_book['title'] = titles
    return df_each_row_book

def calc_weighted_ratings(vote_count, vote_avg, min_votes, vote_avg_all_data):
    """
    Function that computes the weighted rating of each book

    :param vote_count: number of votes for the book
    :param vote_avg: avg vote for the book
    :param min_votes: minimal number of votes for count this book
    :param vote_avg_all_data: avg vote for all the books
    :return:
    """
    # Calculation based on the IMDB formula
    return (vote_count / (vote_count + min_votes) * vote_avg) + (min_votes/(min_votes + vote_count) * vote_avg_all_data)

def calc_top_k(ratings_df, k, books_df):
    """

    :param ratings_df: rating df
    :param k: number of books to return
    :param books_df: book df
    :return: top k recommended books
    """
    # rows numbers = users id, columns number = books id
    users_book_rating_matrix_df = utils.create_users_book_matrix(ratings_df)
    users_book_rating_matrix_df = normalize_ratings(users_book_rating_matrix_df)
    # Calculate mean of all votes
    vote_avg_all_data = users_book_rating_matrix_df.mean().mean()
    # Calculate the minimum number of votes required to be in the chart- m
    sizes_of_groups = ratings_df.groupby('book_id').size()
    min_votes = sizes_of_groups.quantile(0.90)
    # add column of vote_count (now the matrix will contain books X (users +vote count) )
    users_book_rating_matrix_df = users_book_rating_matrix_df.T
    users_book_rating_matrix_df['vote_count'] = sizes_of_groups
    # remove all the votes that doesn't have enough votes
    # rows numbers - book id, columns number- user id and vote_count
    users_book_rating_matrix_df = users_book_rating_matrix_df[users_book_rating_matrix_df['vote_count'] >= min_votes]
    sizes_of_groups = sizes_of_groups[sizes_of_groups >= min_votes]
    # add vote_avg
    vote_avg_per_book = users_book_rating_matrix_df.mean(axis=1)
    weighted_ratings = calc_weighted_ratings(sizes_of_groups, vote_avg_per_book, min_votes, vote_avg_all_data)
    # add score and id to dataframe
    users_book_rating_matrix_df['score'] = weighted_ratings
    users_book_rating_matrix_df['book_id'] = users_book_rating_matrix_df.index
    # add titles according to book_id
    users_book_rating_matrix_df = add_title_column(users_book_rating_matrix_df, books_df)
    # find the top k books with the highest score
    users_book_rating_matrix_df = users_book_rating_matrix_df.sort_values('score', ascending=False)
    # print(users_book_rating_matrix_df[['book_id', 'title', 'score']].head(k))
    top_k_titles = users_book_rating_matrix_df['title'].head(k)
    return top_k_titles

def get_simply_recommendation(k):
    """
    return the top k books - non personalized recommendation

    :param k: number of books to return
    :return: top k books
    """

    # load data
    ratings_df = data_loaders.load_ratings_pd()
    books_df = data_loaders.load_books()
    top_k_titles = calc_top_k(ratings_df, k, books_df)
    return top_k_titles.values

def filter_according_to_place(user_id_book_id_rate, users_data, place):
    """

    :param user_id_book_id_rate: rating df
    :param users_data: users df
    :param place: place to filter
    :return: user_id_book_id_rate dataframe filtered_by place
    """
    users_id_in_place = users_data[users_data['location'] == place]['user_id']
    user_id_book_id_rate_filtered_by_place = user_id_book_id_rate[user_id_book_id_rate['user_id'].isin(users_id_in_place)]
    return user_id_book_id_rate_filtered_by_place

def filter_according_to_age(user_id_book_id_rate, users_data, age):
    """

    :param user_id_book_id_rate: ratings df
    :param users_data: users df
    :param age: age to filter
    :return: user_id_book_id_rate dataframe filtered_by place
    """
    users_id_in_age = users_data[users_data['age'] == age]['user_id']
    user_id_book_id_rate_filtered_by_age = user_id_book_id_rate[user_id_book_id_rate['user_id'].isin(users_id_in_age)]
    return user_id_book_id_rate_filtered_by_age


def get_simply_place_recommendation(place, k):
    """
    Return top k books recommended in place
    :param place: place to filter
    :param k: number of books to return
    :return:  top k books
    """
    ratings_df = data_loaders.load_ratings_pd()
    books_df = data_loaders.load_books()
    users_data = data_loaders.load_users()
    ratings_df = filter_according_to_place(ratings_df, users_data, place)
    top_k_titles = calc_top_k(ratings_df, k, books_df)
    return top_k_titles.values

def get_simply_age_recommendation(age, k):
    """
    Return top k books recommended to people in age x1-y0 (for example for age = 65 - recommendation for age 61-90)
    :param age: age to filter
    :param k: k books to return
    :return: top k books according to age
    """
    ratings_df = data_loaders.load_ratings_pd()
    books_df = data_loaders.load_books()
    users_data = data_loaders.load_users()
    ratings_df = filter_according_to_age(ratings_df, users_data, age)
    top_k_titles = calc_top_k(ratings_df, k, books_df)
    return top_k_titles.values


