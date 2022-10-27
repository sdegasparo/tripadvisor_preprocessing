import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect('hotel_swiss.db')

    df_hotels = pd.read_sql_query("SELECT * FROM hotels", conn)
    df_reviewers = pd.read_sql_query("SELECT * FROM reviewers", conn)
    df_reviews = pd.read_sql_query("SELECT * FROM reviews", conn)

    df_hotels.to_csv('hotels.csv', index=False)
    df_reviewers.to_csv('reviewers.csv', index=False)
    df_reviews.to_csv('reviews.csv', index=False)

    # Rename columns and merge
    df_hotels = df_hotels.rename(columns={'average_score': 'hotel_average_score', 'median_score': 'hotel_median_score',
                              'number_of_reviews': 'hotel_number_of_reviews', 'max_review_one_day': 'hotel_max_review_one_day',
                              'first_review': 'hotel_first_review'})

    df_reviewers = df_reviewers.rename(columns={'number_of_reviews': 'reviewer_number_of_reviews', 'average_score': 'reviewer_average_score',
                                 'median_score': 'reviewer_median_score', 'first_review_date': 'reviewer_first_review_date',
                                 'last_review_date': 'reviewer_last_review_date'})

    df_reviews = df_reviews.rename(columns={'score': 'review_score', 'score_deviation': 'review_score_deviation'})

    df_reviews_reviewers = df_reviews.merge(df_reviewers, left_on='username_id', right_on='username_id')
    df = df_reviews_reviewers.merge(df_hotels, left_on='hotel_id', right_on='hotel_id')

    df.to_csv('tripadvisor.csv', index=False)

if __name__ == '__main__':
    main()
