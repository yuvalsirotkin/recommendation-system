import pandas as pd
import numpy as np

def load_books():
    """

    :return: books dataframe
    """
    books_data = pd.read_csv('books.csv',  low_memory=False, encoding = "ISO-8859-1")
    return books_data

def load_ratings():
    """

    :return: ratings numpy array
    """
    rating_data = np.genfromtxt('ratings.csv', delimiter=',')
    return rating_data

def load_ratings_pd():
    """

    :return: ratings dataframe
    """
    rating_data = pd.read_csv('ratings.csv', low_memory=False, encoding = "ISO-8859-1")
    return rating_data

def load_users():
    """

    :return: users dataframe
    """
    users_data = pd.read_csv('users.csv', low_memory=False, encoding = "ISO-8859-1")
    return users_data

# def load_tags():
#     """
#
#     :return:
#     """
#     tags_data = pd.read_csv('tags.csv', low_memory=False, encoding = "ISO-8859-1")
#     return tags_data

def load_books_tags():
    """

    :return: books and tags for each book - dataframe
    """
    books_tags_data = pd.read_csv('books_tags.csv', low_memory=False, encoding = "ISO-8859-1")
    return books_tags_data

def load_tests():
    """

    :return: test ratings dataframe
    """
    books_tags_data = pd.read_csv('test.csv', low_memory=False, encoding = "ISO-8859-1")
    return books_tags_data