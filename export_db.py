import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect('hotel_murten_1.db')

    df_hotels = pd.read_sql_query("SELECT * FROM hotels", conn)
    df_reviewers = pd.read_sql_query("SELECT * FROM reviewers", conn)
    df_reviews = pd.read_sql_query("SELECT * FROM reviews", conn)

    df_reviews_reviewers = df_reviews.merge(df_reviewers, left_on='username_id', right_on='username_id')
    df = df_reviews_reviewers.merge(df_hotels, left_on='hotel_id', right_on='hotel_id')

    #df.to_csv('hotels.csv', index=False)

    df_hotels.to_csv('hotels1.csv', index=False)
    df_reviewers.to_csv('reviewers1.csv', index=False)
    df_reviews.to_csv('reviews1.csv', index=False)

if __name__ == '__main__':
    main()
