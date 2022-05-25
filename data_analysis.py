import database as db

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


def plot_distribution(df: DataFrame, column: str):
    data = get_distribution(df, column)
    labels = data.index.values
    counts = data.values
    plt.bar(labels, counts, align='center', width=0.3)
    plt.gca().set_xticks(labels)
    plt.show()


def get_distribution(df: DataFrame, column: str):
    return df[column].value_counts()


def get_median(df: DataFrame, column: str):
    return df[column].median()


def get_mean(df: DataFrame, column: str):
    return df[column].mean()


def get_number_of_reviewers_with_multiple_reviews_on_same_hotel(df: DataFrame):
    return df[df['max_reviews_on_same_hotel'] >= 2].count()['username_id']


def main():
    # Get Data
    hotels = pd.DataFrame(db.get_hotels())
    reviewers = pd.DataFrame(db.get_reviewers())
    reviews = pd.DataFrame(db.get_reviews())

    # Change column name
    hotels.columns = ['hotel_id', 'average_score', 'median_score', 'number_of_reviews', 'deviation',
                      'max_review_one_day', 'distortion', 'good_rating_one_day', 'bad_rating_one_day']
    reviewers.columns = ['username_id', 'user_location', 'user_register_date', 'number_of_reviews', 'maximum_reviews',
                         'helpful_vote', 'number_of_good_reviews', 'number_of_bad_reviews', 'average_score',
                         'median_score', 'average_text_characters', 'average_text_sentences', 'max_similarity',
                         'first_review_date', 'last_review_date', 'max_reviews_on_same_hotel']
    reviews.columns = ['review_id', 'username_id', 'hotel_id', 'review_date', 'date_of_stay', 'score', 'title', 'text',
                       'title_length', 'text_length', 'text_sentences', 'text_digits', 'text_uppercase',
                       'text_sentiment', 'text_max_cosine_similarity', 'text_different_tokens',
                       'text_description_similarity', 'hotel_mention', 'score_deviation']

    # TODO: Possibly write the evaluation into a file
    # Print hotel evaluation
    print('************************* Hotel *************************')
    print('Number of hotels:', len(hotels))
    print('Hotel median score:', get_median(hotels, 'average_score'))
    print('Hotel mean score:', get_mean(hotels, 'average_score'))
    plot_distribution(hotels, 'average_score')
    print('Hotel median reviews:', get_median(hotels, 'number_of_reviews'))
    print('Hotel mean reviews:', get_mean(hotels, 'number_of_reviews'))

    # Print Reviewers evaluation
    print('\n************************* Reviewers *************************')
    print('Number of reviewers:', len(reviewers))
    print('Reviewers median reviews', get_median(reviewers, 'number_of_reviews'))
    print('Reviewers mean reviews', get_mean(reviewers, 'number_of_reviews'))
    plot_distribution(reviewers, 'number_of_reviews')
    print('Reviewers with multiple reviews on same hotel', get_number_of_reviewers_with_multiple_reviews_on_same_hotel(reviewers))

    # Print Reviews evaluation
    print('\n************************* Reviews *************************')
    print('Number of reviews:', len(reviews))
    print('Reviews median score:', get_median(reviews, 'score'))
    print('Reviews mean score:', get_mean(reviews, 'score'))
    plot_distribution(reviews, 'score')
    print('Reviews median sentiment:', get_median(reviews, 'text_sentiment'))
    print('Reviews mean sentiment:', get_mean(reviews, 'text_sentiment'))
    print('Reviews median characters:', get_median(reviews, 'text_length'))
    print('Reviews mean characters:', get_mean(reviews, 'text_length'))
    print('Reviews median sentences:', get_median(reviews, 'text_sentences'))
    print('Reviews mean sentences:', get_mean(reviews, 'text_sentences'))
    plot_distribution(reviews, 'text_sentences')


if __name__ == '__main__':
    main()
