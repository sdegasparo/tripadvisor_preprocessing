import json
import re
import datetime

from transformers import pipeline

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


def month_year_to_date(date: str) -> datetime:
    """
    Convert register date or date of stay string from raw data to datetime
    :param date: str
    :return: date: datetime

    >>> month_year_to_date('4.2020')
    2020-04-01

    >>> month_year_to_date('10.2004')
    2004-10-01
    """
    date_array = date.split('.')
    day = 1
    month = int(date_array[0])
    year = int(date_array[1])

    return datetime.date(year, month, day)


def day_month_year_to_date(date: str) -> datetime:
    """
    Convert review date string from raw data to datetime
    :param date: str
    :return: date: datetime

    >>> day_month_year_to_date('26.10.2015')
    2015-10-26

    >>> day_month_year_to_date('2.5.1999')
    1999-05-02
    """
    date_array = date.split('.')
    day = int(date_array[0])
    month = int(date_array[1])
    year = int(date_array[2])

    return datetime.date(year, month, day)


def get_reviews_by_hotel_id(df: DataFrame, hotel_id: str) -> DataFrame:
    """

    :param df: DataFrame
    :param hotel_id: str
    :return: All reviews from this hotel: DataFrame
    """
    return df.loc[df['hotel_id'] == hotel_id]


def get_reviews_by_username_id(df: DataFrame, username_id: str) -> DataFrame:
    """

    :param df: DataFrame
    :param username_id: str
    :return: All reviews from this user: DataFrame
    """
    return df.loc[df['username_id'] == username_id]


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


def get_number_of_different_tokens(review: str) -> int:
    """
    :param review: str
    :return: number of different tokens

    >>> get_number_of_different_tokens('This my the 3 rd time in this hotel. The service is very NICE. They know me and know my tastes. I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area. They serve you with a smile and this is so nice !')
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


def remove_stopwords(text: str) -> set:
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = {token for token in tokens if not token in stopwords}

    return filtered_tokens


def clean_string(text: str) -> str:
    exclist = string.punctuation
    table_ = str.maketrans(exclist, ' ' * len(exclist))
    text = ' '.join(text.translate(table_).split())
    return text


def get_cosine_similarity(text_1: str, text_2):
    """
    :param text_1: str
    :param text_2: str
    :return: cosine similarity float or bool

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


def get_max_cosine_similarity_hotel(df: DataFrame, hotel_id: str, review_id: str, review_text: str) -> float:
    """
    :param df: DataFrame
    :param hotel_id: str
    :param review_id: str
    :param review_text: str
    :return: Maximum cosine similarity score: float

    >>> df_cosine = pd.DataFrame({'hotel_id': ['42', '42', '42', '42'], 'review_id': [1, 2, 3, 4], 'review_text': ['The hotel Hilton was nice', 'The hotel was nice', 'I did not liked the Hilton hotel', 'The best Hilton in Switzerland']})
    >>> get_max_cosine_similarity_hotel(df_cosine, '42', '1', 'The hotel Hilton was nice')
    0.8944
    """
    df = get_reviews_by_hotel_id(df, hotel_id)
    max_similarity = 0
    for index, row in df.iterrows():
        if int(row['review_id']) == int(review_id):
            continue

        similarity = get_cosine_similarity(review_text, row['review_text'])
        if max_similarity < similarity:
            max_similarity = similarity

    return max_similarity


# def get_sentiment(review: str) -> float:
#     """
#     :param review: str
#     :return: the sentiment of the review
#     """
#     sentiment = sentiment_model(review)
#     if sentiment[0]['label'] == 'Negative':
#         return round(- sentiment[0]['score'], 4)
#     else:
#         return round(sentiment[0]['score'], 4)


# Reviewer specific
def get_number_of_reviews_for_reviewer(df: DataFrame) -> int:
    """
    :param df: DataFrame
    :return: sum of reviews by username: int
    """
    return len(df)


def get_max_reviews_on_one_day(df: DataFrame) -> int:
    """
    :param df: DataFrame
    :return: Maximum number of reviews in one day: int
    """
    return int(df.groupby(['review_date']).size().max())


def get_sum_of_helpful_votes(df: DataFrame) -> int:
    """
    :param df: DataFrame
    :return: sum of helpful votes: int
    """
    return df['review_helpful_vote'].sum()


def get_number_of_good_rating(df: DataFrame) -> float:
    """
    :param df: DataFrame
    :return: percentage of good reviews by username: float
    """
    number_of_reviews = len(df)
    number_of_good_reviews = len(df.loc[df['review_score'] >= 4])
    percentage = number_of_reviews / number_of_good_reviews
    if np.isnan(percentage):
        return 0

    return percentage


def get_number_of_bad_rating(df: DataFrame) -> float:
    """
    :param df: DataFrame
    :return: percentage of bad reviews by username: float
    """
    number_of_reviews = len(df)
    number_of_good_reviews = len(df.loc[df['review_score'] <= 2])
    percentage = number_of_reviews / number_of_good_reviews
    if np.isnan(percentage):
        return 0

    return percentage


def get_average_score(df: DataFrame) -> float:
    """
    :param df: DataFrame
    :return: Average rating score of reviewer: float
    """
    return df['review_score'].mean()


def get_average_text_characters(df: DataFrame) -> float:
    """
    :param df: DataFrame
    :return: Average text character of reviewer
    """
    number_of_reviews = len(df)
    character_sum = 0
    for index, row in df.iterrows():
        character_sum += get_number_of_characters(row['review_text'])

    return character_sum / number_of_reviews


def get_average_text_sentences(df: DataFrame) -> float:
    """
    :param df: DataFrame
    :return: Average text sentences of reviewer
    """
    number_of_reviews = len(df)
    sentence_sum = 0
    for index, row in df.iterrows():
        sentence_sum += get_number_of_sentences(row['review_text'])

    return sentence_sum / number_of_reviews


def get_date_of_first_review(df: DataFrame) -> datetime:
    """
    :param df: DataFrame
    :return: date of first review: datetime
    """
    return df['review_date'].min()


def get_date_of_last_review(df: DataFrame) -> datetime:
    """
    :param df: DataFrame
    :return: date of last review: datetime
    """
    return df['review_date'].max()


def get_max_reviews_on_same_hotel(df: DataFrame) -> int:
    """
    :param df: DataFrame
    :return: Maximum reviews on same hotel: int
    """
    df = df.sort_values(by=['hotel_id'], ascending=True)
    count = 0
    id = None
    for index, row in df.iterrows():
        if id == row['hotel_id']:
            count += 1
        else:
            id = row['hotel_id']
            count = 1

    return count


def get_max_cosine_similarity_reviewer(df: DataFrame):
    """
    :param df: DataFrame
    :return: Maximum cosine similarity score: float
    """
    max_similarity = 0
    for index_outer, row_outer in df.iterrows():
        review_outer = row_outer['review_text']
        for index_inner, row_inner in df.iterrows():
            if index_outer == index_inner:
                continue

            similarity = get_cosine_similarity(review_outer, row_inner['review_text'])
            if max_similarity < similarity:
                max_similarity = similarity

    return max_similarity


# Hotel specific
def get_number_of_reviews_by_hotel_id(df: DataFrame, hotel_id: str) -> int:
    """
    Returns the number of all reviews

    :param df: DataFrame
    :param hotel_id: str
    :return: sum of reviews: int
    """
    return len(get_reviews_by_hotel_id(df, hotel_id))


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
    df = get_reviews_by_hotel_id(df, hotel_id)
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


def get_hotel_score_distortion(df: DataFrame, hotel_id: str):
    """
    Calculate the difference between hotel score and calculated hotel score of a random subset
    :param df: DataFrame
    :param hotel_id: str
    :return: The absolute distortion: float
    """
    hotel_score = df.loc[df['hotel_id'] == hotel_id]['hotel_score'].values[0]
    sum_score = 0
    experiments = 10
    for i in range(experiments):
        df_hotel = df.loc[df['hotel_id'] == hotel_id]['review_score']
        review_scores = df_hotel.drop(df.sample(frac=0.2).index)
        sum_score = abs(hotel_score - np.mean(review_scores))

    return sum_score / experiments


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
        # text_sentiment = get_sentiment(text)
        text_max_cosine_similarity = get_max_cosine_similarity_hotel(df, hotel_id, review_id, text)
        text_different_tokens = get_number_of_different_tokens(text)
        text_description_similarity = get_cosine_similarity(text, row['hotel_description'])
        hotel_mention = get_number_of_hotel_name_mention(row['hotel_name'], text)
        score_deviation = get_deviation_from_rating(row['hotel_score'], score)


def db_insert_reviewer(df):
    df = df.sort_values(by=['username_id'], ascending=True)
    id = None
    for index, row in df.iterrows():
        username_id = row['username_id']
        if id is not username_id:
            df_username = get_reviews_by_username_id(df, username_id)
            id = username_id
            user_location = row['user_location']
            user_register_date = row['user_register_date']
            number_of_reviews = get_number_of_reviews_for_reviewer(df_username)
            maximum_reviews = get_max_reviews_on_one_day(df_username)
            helpful_vote = get_sum_of_helpful_votes(df_username)
            number_of_good_reviews = get_number_of_good_rating(df_username)
            number_of_bad_reviews = get_number_of_bad_rating(df_username)
            average_score = get_average_score(df_username)
            average_text_characters = get_average_text_characters(df_username)
            average_text_sentences = get_average_text_sentences(df_username)
            max_similarity = get_max_cosine_similarity_reviewer(df_username)
            # deviation = Percentage of deviation between other hotel reviews
            first_review_date = get_date_of_first_review(df_username)
            last_review_date = get_date_of_last_review(df_username)
            max_reviews_on_same_hotel = get_max_reviews_on_same_hotel(df_username)


def db_insert_hotel(df):
    # df = df.reset_index()  # Not sure if it's needed
    df = df.sort_values(by=['hotel_id'], ascending=True)
    id = None
    for index, row in df.iterrows():
        hotel_id = row['hotel_id']
        if id is not hotel_id:
            id = hotel_id
            hotel_id = hotel_id
            score = row['hotel_score']
            number_of_reviews = get_number_of_reviews_by_hotel_id(df, hotel_id)
            deviation = get_hotel_score_deviation(df, hotel_id)
            max_review_one_day = get_max_review_percentage_on_one_day(df, hotel_id, number_of_reviews)
            distortion = get_hotel_score_distortion(df, hotel_id)
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
    # raw_data = load_json('tripadvisor_good.json')
    raw_data = load_json('schweiz_5min.json')

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

    # Drop NaN
    df_user_review = df_user_review.dropna(subset=['ur_review_id'])

    # Covert Date String to Dates
    for index, row in df_user.iterrows():
        row['u_user_register_date'] = month_year_to_date(row['u_user_register_date'])

    for index, row in df_user_review.iterrows():
        row['ur_review_date'] = day_month_year_to_date(row['ur_review_date'])

        # Check for NaN values
        date_of_stay = row['ur_date_of_stay']
        if not pd.isna(date_of_stay):
            row['ur_date_of_stay'] = month_year_to_date(date_of_stay)

    # Merge DataFrames to one big DataFrame
    df_hotel_hotel_review = pd.merge(df_hotel, df_hotel_review, left_on='h_hotel_id', right_on='hr_hotel_id')
    df_user_user_review = pd.merge(df_user, df_user_review, left_on='u_username_id', right_on='ur_username_id')
    df = pd.merge(df_hotel_hotel_review, df_user_user_review, left_on='hr_review_id', right_on='ur_review_id')
    df = df.drop(columns=['hr_hotel_id', 'ur_username_id', 'ur_review_id'])
    # print(df)
    df = df.rename(columns={
        'h_hotel_id': 'hotel_id',
        'h_hotel_name': 'hotel_name',
        'h_hotel_score': 'hotel_score',
        'h_hotel_description': 'hotel_description',
        'hr_review_id': 'review_id',
        'u_username_id': 'username_id',
        'u_user_location': 'user_location',
        'u_user_register_date': 'user_register_date',
        'ur_review_helpful_vote': 'review_helpful_vote',
        'ur_review_date': 'review_date',
        'ur_date_of_stay': 'date_of_stay',
        'ur_review_score': 'review_score',
        'ur_review_title': 'review_title',
        'ur_review_text': 'review_text'
    })

    df_user_user_review = df_user_user_review.rename(columns={
        'u_username_id': 'username_id',
        'u_user_location': 'user_location',
        'u_user_register_date': 'user_register_date',
        'ur_review_helpful_vote': 'review_helpful_vote',
        'ur_review_date': 'review_date',
        'ur_date_of_stay': 'date_of_stay',
        'ur_review_score': 'review_score',
        'ur_review_title': 'review_title',
        'ur_review_text': 'review_text'
    })

    # print(len(df_hotel))
    # print(df)

    # print(df_hotel_hotel_review)
    # print(df_user_user_review)
    # print(df_hotel)
    # print(df)
    # df = df.sort_values(by=['hotel_id']) Not sure if it's needed
    # print(df)

    # Database
    # db_insert_hotel(df)
    # db_insert_reviews(df_user_user_review)
    # db_insert_reviewer(df)

    # TESTS

    # df_cosine = pd.DataFrame({
    #     'hotel_id': ['42', '42', '42', '42'],
    #     'review_id': [1, 2, 3, 4],
    #     'review_text': ['The hotel Hilton was nice', 'The hotel was nice', 'I did not liked the Hilton hotel', 'The best Hilton in Switzerland']
    # })
    # print(get_max_cosine_similarity_hotel(df_cosine, '42', '1', 'The hotel Hilton was nice'))

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

    # sentiment_pipeline = pipeline("sentiment-analysis")
    # sentiment_model = pipeline(model="Tobias/bert-base-german-cased_German_Hotel_sentiment")

    main()
