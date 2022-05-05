import json
import pandas as pd
from pandas import DataFrame
import numpy as np

import string

import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from typing import List


def load_json(file: str):
    f = open(file)
    data = json.load(f)
    f.close()

    return data


# Review specific
def get_number_of_characters(review: str) -> int:
    """
    :param review: str
    :return: number of character: int

    >>> get_number_of_characters('Nice room!')
    10
    """
    return len(review)


def get_number_of_sentences(review: str) -> int:
    """
    :param review: str
    :return: number of sentences: int

    >>> get_number_of_sentences('This my the 3 rd time in this hotel. The service is very NICE. \\
                                They know me and know my tastes. \\
                                I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. \\
                                They serve you with a smile and this is so nice !')
    5
    """
    return len(sent_tokenize(review))


def get_number_of_different_token(review: str) -> int:
    """

    :param review:
    :return:

    >>> get_number_of_different_token('This my the 3 rd time in this hotel. The service is very NICE. \\
                                They know me and know my tastes. \\
                                I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. \\
                                They serve you with a smile and this is so nice !')
    41
    """
    return len(set(word_tokenize(review)))


def get_percentage_of_digit(review: str) -> float:
    """

    :param review: str
    :return: percentage of digits: float
    >>> get_percentage_of_digit('This my the 3 rd time in this hotel. The service is very NICE. \\
                                They know me and know my tastes. \\
                                I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. \\
                                They serve you with a smile and this is so nice !')
    0.002777777777777778
    """
    total_characters = get_number_of_characters(review)
    digits = sum(c.isdigit() for c in review)
    return digits / total_characters


def get_percentage_of_uppercase_words(review: str) -> float:
    """

    :param review: str
    :return: percentage of uppercase words: float

    >>> get_percentage_of_uppercase_words('This my the 3 rd time in this hotel. The service is very NICE. \\
                                They know me and know my tastes. \\
                                I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. \\
                                They serve you with a smile and this is so nice !')
    0.05084745762711865
    """
    tokens = word_tokenize(review)
    total_words = len(tokens)
    uppercase_words = sum(token.isupper() for token in tokens)
    return uppercase_words / total_words


def get_number_of_hotel_name_mention(hotel_name: str, review: str) -> int:
    token_frequency = FreqDist(review)
    return token_frequency[hotel_name]


def get_deviation_from_rating(hotel_score: float, review_score: int) -> float:
    """
    :param hotel_score: float
    :param review_score: int
    :return: deviation: float

    >>> get_deviation_from_rating(4.5, 2)
    2.5

    >>> get_deviation_from_rating(2, 5)
    3
    """
    return abs(hotel_score - review_score)


def remove_stopwords(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = {token for token in tokens if not token in stopwords}

    return filtered_tokens


def clean_string(text: str) -> str:
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])

    return text


def get_cosine_similarity(text_1: str, text_2: str) -> float:
    text_1 = clean_string(text_1)
    text_2 = clean_string(text_2)

    data = []
    data.append(text_1)
    data.append(text_2)

    vectorizer = CountVectorizer().fit_transform(data)
    vectors = vectorizer.toarray()

    # Just return the cosine similiarity of the two texts and not the whole matrix
    return cosine_similarity(vectors)[0, 1]


# Reviewer specific
def get_sum_of_reviews(df: DataFrame):
    pass


def get_sum_of_same_day_reviews():
    pass


# Hotel specific
def get_hotel_score_deviation(df: DataFrame, hotel_id: str):
    """
    Compute the standard deviation of all reviews scores of a specific hotel

    :param df: DataFrame
    :param hotel_id: str
    :return: standard deviation: float
    """
    all_review_scores_from_hotel_id = df.loc[df['hotel_id'] == hotel_id]['review_score']
    return np.std(all_review_scores_from_hotel_id)


def get_max_review_on_one_day(df: DataFrame, hotel_id: str):
    """
    Calculate the maximum number of ratings in percent on one day

    :param df: DataFrame
    :param hotel_id: str
    :return: maximum number of ratings in one day in percent: float
    """
    df = df.loc[df['hotel_id'] == hotel_id]
    number_of_reviews = float(df['hotel_number_of_reviews'].values[0])
    max_review_on_one_day = int(df.groupby(['review_date']).size().max())
    return max_review_on_one_day / number_of_reviews


def get_number_good_rating_on_one_day(df: DataFrame):
    df = df.groupby(['hotel_id', 'ur_review_date'])
    return df.loc[df['review_score'] >= 4].size().max()


def check_distortion(df: DataFrame):
    df = df.groupby(['hr_hotel_id'])
    df = df.drop(df.sample(frac=0.8).index)
    return


def db_insert_hotel(df):
    # df = df.reset_index()  # Not sure if it's needed
    for index, row in df.iterrows():
        hotel_id = row['hotel_id']
        score = row['hotel_score']
        number_of_reviews = row['hotel_number_of_reviews']
        deviation = get_hotel_score_deviation(df, hotel_id)  # Change
        max_review_one_day = get_max_review_on_one_day(df, hotel_id)
        print(hotel_id, score, number_of_reviews, deviation, max_review_one_day)


def main():
    # pd.set_option('display.max_columns', None)
    raw_data = load_json('tripadvisor.json')

    hotel = []
    hotel_review = []
    user = []
    user_review = []

    # Check for Reviews first, because it has more reviews. Should be faster
    for data in raw_data:
        if 'ur_username_id' in data:
            user_review.append(data)
        elif 'u_username_id' in data:
            user.append(data)
        elif 'hr_hotel_id' in data:
            hotel_review.append(data)
        elif 'h_hotel_id' in data:
            hotel.append(data)

    # Create DataFrames
    df_hotel = pd.DataFrame(hotel).drop_duplicates()
    df_hotel_review = pd.DataFrame(hotel_review).drop_duplicates()
    df_user = pd.DataFrame(user).drop_duplicates()
    df_user_review = pd.DataFrame(user_review).drop_duplicates()

    # print(df_hotel)

    # Merge DataFrames to one big DataFrame
    df_hotel_hotel_review = pd.merge(df_hotel, df_hotel_review, left_on='h_hotel_id', right_on='hr_hotel_id')
    df_user_user_review = pd.merge(df_user, df_user_review, left_on='u_username_id', right_on='ur_username_id')
    df = pd.merge(df_hotel_hotel_review, df_user_user_review, left_on='hr_review_id', right_on='ur_review_id')
    df = df.drop(columns=['hr_hotel_id', 'ur_username_id', 'ur_review_id'])
    # print(df)
    df.columns = ['hotel_id', 'hotel_name', 'hotel_score', 'hotel_number_of_reviews', 'hotel_description',
                  'review_id', 'username_id', 'user_location', 'user_register_date', 'review_helpful_vote',
                  'review_date', 'date_of_stay', 'review_score', 'review_title', 'review_text']

    # print(df_hotel_hotel_review)
    # print(df_user_user_review)
    # print(df)

    # Database
    db_insert_hotel(df)

    # TESTS

    text = """This my the 3 rd time in this hotel. The service is very NICE.
                                They know me and know my tastes.
                                I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area.
                                They serve you with a smile and this is so nice !"""
    # print(get_number_of_characters(text))
    # print(get_number_of_sentences(text))
    # print(get_number_of_different_token(text))
    # print(get_percentage_of_digit(text))
    # print(get_percentage_of_uppercase_words(text))
    # print(get_number_of_hotel_name_mention('I', text))
    # print(get_deviation_from_rating(4.5, 2))
    # print(get_deviation_from_rating(2, 5))
    # print(remove_stopwords('Das ist ein Test'))
    # print(get_cosine_similarity('Das ist ein komischer Test heute von Frau Dernd',
    #                             'Was wollen wir heute machen an diesen schönen Tag?'))

    # print(df_user.loc[df_user['u_username_id'] == 'Ursli2'])
    #
    # print(len(df_hotel.loc[df_hotel['h_hotel_id'] == '641393']))

    # print(df_hotel.groupby(['h_hotel_id']).count())
    # print(df_user.groupby(['u_username_id']).count())
    #
    # number_review = [{'hr_hotel_id': '1', 'ur_review_date': '5.3.2022'},
    #                  {'hr_hotel_id': '2', 'ur_review_date': '5.3.2022'},
    #                  {'hr_hotel_id': '1', 'ur_review_date': '5.3.2022'},
    #                  {'hr_hotel_id': '1', 'ur_review_date': '2.3.2022'},
    #                  {'hr_hotel_id': '2', 'ur_review_date': '3.3.2022'}]
    # df = pd.DataFrame(number_review)
    # print('Number Reviews', get_max_review_on_one_day(df))


if __name__ == '__main__':
    stopwords = stopwords.words('german')
    main()
