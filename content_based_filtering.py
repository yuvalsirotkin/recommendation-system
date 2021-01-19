#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import data_loaders
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import utils
from sklearn.metrics.pairwise import pairwise_distances
from ast import literal_eval

books_df = data_loaders.load_books()
ratings_df = data_loaders.load_ratings_pd()
users_books_ratings_matrix_pd = utils.create_users_book_matrix(ratings_df)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x)
        else:
            return ''

features = ['authors', 'original_publication_year', 'original_title', 'language_code']
for feature in features:
    books_df[feature] = books_df[feature].apply(clean_data)


def create_item_sim_according_to_ratings():
    sim= 'cosine'
    # normalize users_books_ratings_matrix_pd matrix
    mean_user_rating = users_books_ratings_matrix_pd.mean(axis= 1).to_numpy().reshape(-1,1)
    ratings_normal = (users_books_ratings_matrix_pd - mean_user_rating)
    ratings_normal[np.isnan(ratings_normal)] = 0
    raitingItem = ratings_normal
    # calc item similarity according to ratings of users (I1 similar to I2 if it gets similar ratings from users)
    item_similarity = 1 - pairwise_distances(raitingItem.T, metric=sim)
    return item_similarity

def cretae_groups_of_tags(books_tags_df, books_df):
    gb= books_tags_df.groupby('goodreads_book_id')
    ids = gb.groups.keys()
    for id in ids:
        tags = gb.get_group(id)['tag_id']


def create_soup(x, groups):
    # get tags
    good_id = x['goodreads_book_id']
    tags = ''
    if good_id in groups.groups:
        tags = groups.get_group(good_id)['tag_id'].values
        tags = np.array2string(tags).replace('\n', '').replace('[', '').replace(']', '')
    return str(x['title'])  + " " +str(x['authors']) + " " + str(x['original_publication_year'])+ " " + str(x['language_code']) + " " + tags



def build_contact_sim_metrix():
    global books_df
    books_tags_df = data_loaders.load_books_tags()
    gb = books_tags_df.groupby('goodreads_book_id')
    books_df['soup'] = books_df.apply(create_soup, groups=gb, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(books_df['soup'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    item_similarity = create_item_sim_according_to_ratings()
    sim_matrix = 0.5 * cosine_sim2 + 0.5 * item_similarity
    # books_df = books_df.reset_index()
    # indices = pd.Series(books_df.index, index=books_df['title'])
    # print(indices[:10])
    return sim_matrix


def get_contact_recommendation(title, k):
    cosine_sim = build_contact_sim_metrix()
    # Get the index of the movie that matches the title
    global books_df
    books_df = books_df.reset_index()
    indices = pd.Series(books_df.index, index=books_df['original_title'])
    # print(indices[:10])
    idx = indices[str.lower(title)]
    if isinstance(idx, pd.Series):
        idx = idx[0]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies (the first is the movie we asked)
    sim_scores = sim_scores[1:k+1]
    # Get the movie indices
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return books_df['title'].iloc[book_indices]

# def get_contact_recommendation(book_name, k):
#     cosine_sim = build_contact_sim_metrix()
#     get_recommendations(book_name, cosine_sim=cosine_sim)

