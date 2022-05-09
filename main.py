import json
import re

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


def get_hotel_description_by_hotel_id(df: DataFrame, hotel_id: str):
    """
    Return the hotel description if it's exists
    :param df: DataFrame
    :param hotel_id: str
    :return: hotel_description: str or False
    """
    hotel_description = df.loc[df['hotel_id'] == hotel_id]['hotel_description']
    if hotel_description.any():
        return hotel_description.values[0]

    return False


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

    >>> get_number_of_sentences('This my the 3 rd time in this hotel. The service is very NICE. They know me and know my tastes. I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. They serve you with a smile and this is so nice !')
    5
    """
    return len(sent_tokenize(review))


def get_number_of_different_token(review: str) -> int:
    """

    :param review:
    :return:

    >>> get_number_of_different_token('This my the 3 rd time in this hotel. The service is very NICE. They know me and know my tastes. I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. They serve you with a smile and this is so nice !')
    41
    """
    return len(set(word_tokenize(review)))


def get_percentage_of_digit(review: str) -> float:
    """

    :param review: str
    :return: percentage of digits: float
    >>> get_percentage_of_digit('This my the 3 rd time in this hotel. The service is very NICE. They know me and know my tastes. I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. They serve you with a smile and this is so nice !')
    0.003787878787878788
    """
    total_characters = get_number_of_characters(review)
    digits = sum(c.isdigit() for c in review)
    return digits / total_characters


def get_percentage_of_uppercase_words(review: str) -> float:
    """

    :param review: str
    :return: percentage of uppercase words: float

    >>> get_percentage_of_uppercase_words('This my the 3 rd time in this hotel. The service is very NICE. They know me and know my tastes. I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. They serve you with a smile and this is so nice !')
    0.05084745762711865
    """
    tokens = word_tokenize(review)
    total_words = len(tokens)
    uppercase_words = sum(token.isupper() for token in tokens)
    return uppercase_words / total_words


def get_number_of_hotel_name_mention(hotel_name: str, review: str) -> int:
    """

    :param hotel_name: str
    :param review: str
    :return: number of hotel mention: int

    >>> get_number_of_hotel_name_mention('Hotel Hilton', 'The Hotel Hilton is the best hotel. It was really nice at the hotel hilton.')
    2
    """
    return len(re.findall(hotel_name.lower(), review.lower()))


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
    exclist = string.punctuation
    table_ = str.maketrans(exclist, ' ' * len(exclist))
    text = ' '.join(text.translate(table_).split())
    return text


def get_cosine_similarity(text_1: str, text_2: str) -> float:
    """

    :param text_1:
    :param text_2:
    :return:

    >>> get_cosine_similarity('This is a Test', 'This is a Test')
    1.0

    >>> get_cosine_similarity('This is a Test', 'No similarity')
    0.0

    >>> get_cosine_similarity('Test', False)
    False
    """
    if text_1 and text_2:
        text_1 = clean_string(text_1)
        text_2 = clean_string(text_2)

        data = []
        data.append(text_1)
        data.append(text_2)

        vectorizer = CountVectorizer().fit_transform(data)
        vectors = vectorizer.toarray()

        # Just return the cosine similiarity of the two texts and not the whole matrix
        return round(cosine_similarity(vectors)[0, 1], 4)
    else:
        return False


# Reviewer specific
def get_sum_of_reviews(df: DataFrame):
    pass


def get_sum_of_same_day_reviews():
    pass


# Hotel specific
def get_number_of_reviews(df: DataFrame, hotel_id: str):
    """
    Returns the number of all reviews

    :param df: DataFrame
    :param hotel_id: str
    :return: sum of reviews: int
    """
    return len(df.loc[df['hotel_id'] == hotel_id])


def get_hotel_score_deviation(df: DataFrame, hotel_id: str):
    """
    Compute the standard deviation of all reviews scores of a specific hotel

    :param df: DataFrame
    :param hotel_id: str
    :return: standard deviation: float
    """
    all_review_scores_from_hotel_id = df.loc[df['hotel_id'] == hotel_id]['review_score']
    return np.std(all_review_scores_from_hotel_id)


def get_max_review_percentage_on_one_day(df: DataFrame, hotel_id: str, number_of_reviews: int):
    """
    Calculate the maximum number of ratings in percent on one day

    :param df: DataFrame
    :param hotel_id: str
    :param number_of_reviews: int
    :return: maximum number of ratings in one day in percent: float
    """
    df = df.loc[df['hotel_id'] == hotel_id]
    max_review_on_one_day = int(df.groupby(['review_date']).size().max())
    return max_review_on_one_day / number_of_reviews


def get_number_good_rating_on_one_day(df: DataFrame, hotel_id: str, number_of_reviews: int):
    """
    Calculate the maximum number of good ratings in one day in percent

    :param df: DataFrane
    :param hotel_id: str
    :param number_of_reviews: int
    :return: maximum number of good ratings in one day in percent: float
    """
    df = df.loc[(df['hotel_id'] == hotel_id) & (df['review_score'] >= 4)]
    max_good_review_on_one_day = df.groupby(['review_date']).size().max()
    percentage = max_good_review_on_one_day / number_of_reviews
    if np.isnan(percentage):
        return 0

    return percentage


def get_number_bad_rating_on_one_day(df: DataFrame, hotel_id: str, number_of_reviews: int):
    """
    Calculate the maximum number of bad ratings in one day in percent

    :param df: DataFrane
    :param hotel_id: str
    :param number_of_reviews: int
    :return: maximum number of bad ratings in one day in percent: float
    """
    df = df.loc[(df['hotel_id'] == hotel_id) & (df['review_score'] <= 2)]
    max_bad_review_on_one_day = df.groupby(['review_date']).size().max()
    percentage = max_bad_review_on_one_day / number_of_reviews
    if np.isnan(percentage):
        return 0

    return percentage


# TODO
def get_hotel_score_distortion(df: DataFrame, hotel_id: str):
    """
    Calculate the difference between hotel score and calculated hotel score of a random subset

    :param df: DataFrame
    :param hotel_id: str
    :return: The absolute distortion: float
    """
    hotel_score = df.loc[df['hotel_id'] == hotel_id]['hotel_score'].values[0]
    df = df.loc[df['hotel_id'] == hotel_id]['review_score']
    review_scores = df.drop(df.sample(frac=0.2).index)
    # return abs(hotel_score - np.mean(review_scores))
    return 0


# Insert data to database
def db_insert_reviews(df):
    for index, row in df.iterrows():
        review_id = row['review_id']
        username_id = row['username_id']
        hotel_id = row['hotel_id']
        review_date = row['review_date']
        date_of_stay = row['date_of_stay']
        score = row['review_score']
        title = row['review_title']
        text = row['review_text']
        title_length = get_number_of_characters(title)
        text_length = get_number_of_characters(text)
        text_sentences = get_number_of_sentences(text)
        text_digits = get_percentage_of_digit(text)
        text_uppercase = get_percentage_of_uppercase_words(text)
        # TODO: Prozentsatz der positiven/negativen meinungsbildenden Wörter in jeder Rezension
        # text_cosine_similarity = get_cosine_similarity()
        text_different_tokens = get_number_of_different_token(text)
        text_description_similarity = get_cosine_similarity(text, row['hotel_description'])
        hotel_mention = get_number_of_hotel_name_mention(row['hotel_name'], text)
        score_deviation = get_deviation_from_rating(row['hotel_score'], score)


def db_insert_reviewer(df):
    id = None
    for index, row in df.iterrows():
        reviewer_id = row['reviewer_id']
        if id is not reviewer_id:
            id = reviewer_id
            # TODO Add the others


def db_insert_hotel(df):
    # df = df.reset_index()  # Not sure if it's needed
    id = None
    for index, row in df.iterrows():
        hotel_id = row['hotel_id']
        if id is not hotel_id:
            id = hotel_id
            hotel_id = hotel_id
            score = row['hotel_score']
            number_of_reviews = get_number_of_reviews(df, hotel_id)
            deviation = get_hotel_score_deviation(df, hotel_id)
            max_review_one_day = get_max_review_percentage_on_one_day(df, hotel_id, number_of_reviews)
            distortion = get_hotel_score_distortion(df, hotel_id)  # TODO
            good_rating_one_day = get_number_good_rating_on_one_day(df, hotel_id, number_of_reviews)
            bad_rating_one_day = get_number_bad_rating_on_one_day(df, hotel_id, number_of_reviews)

            print(hotel_id, score, number_of_reviews, deviation, max_review_one_day, distortion, good_rating_one_day,
                  bad_rating_one_day)
            # TODO: insert into DB
        else:
            # TODO: insert into DB with values already determined
            pass


def main():
    # pd.set_option('display.max_columns', None)
    raw_data = load_json('tripadvisor_good.json')

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

    # Create DataFrames and drop any duplicates
    df_hotel = pd.DataFrame(hotel).drop_duplicates()
    df_hotel_review = pd.DataFrame(hotel_review).drop_duplicates()
    df_user = pd.DataFrame(user).drop_duplicates()
    df_user_review = pd.DataFrame(user_review).drop_duplicates()

    # Merge DataFrames to one big DataFrame
    df_hotel_hotel_review = pd.merge(df_hotel, df_hotel_review, left_on='h_hotel_id', right_on='hr_hotel_id')
    df_user_user_review = pd.merge(df_user, df_user_review, left_on='u_username_id', right_on='ur_username_id')
    df = pd.merge(df_hotel_hotel_review, df_user_user_review, left_on='hr_review_id', right_on='ur_review_id')
    df = df.drop(columns=['hr_hotel_id', 'ur_username_id', 'ur_review_id'])
    # print(df)
    df = df.rename(columns={'h_hotel_id': 'hotel_id', 'h_hotel_name': 'hotel_name', 'h_hotel_score': 'hotel_score',
                            'h_hotel_description': 'hotel_description',
                            'hr_review_id': 'review_id', 'u_username_id': 'username_id',
                            'u_user_location': 'user_location', 'u_user_register_date': 'user_register_date',
                            'ur_review_helpful_vote': 'review_helpful_vote',
                            'ur_review_date': 'review_date', 'ur_date_of_stay': 'date_of_stay',
                            'ur_review_score': 'review_score', 'ur_review_title': 'review_title',
                            'ur_review_text': 'review_text'})

    # print(df_hotel_hotel_review)
    # print(df_user_user_review)
    # print(df_hotel)
    # print(df)
    # df = df.sort_values(by=['hotel_id']) Not sure if it's needed
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
