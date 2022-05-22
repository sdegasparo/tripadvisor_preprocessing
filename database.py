import sqlite3

# Change to database.db file for production
conn = sqlite3.connect(':memory:')
c = conn.cursor()


def create_hotels_table():
    with conn:
        c.execute("""CREATE TABLE IF NOT EXISTS hotels (
            hotel_id INTEGER PRIMARY KEY,
            average_score REAL,
            median_score REAL,
            number_of_reviews INTEGER,
            deviation REAL,
            max_review_one_day REAL,
            distortion REAL,
            good_rating_one_day REAL,
            bad_rating_one_day REAL
        )""")


def create_reviews_table():
    with conn:
        c.execute("""CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY,
            username_id TEXT,
            hotel_id INTEGER,
            review_date TEXT,
            date_of_stay TEXT,
            score REAL,
            title TEXT,
            text TEXT,
            title_length INTEGER,
            text_length INTEGER,
            text_sentences INTEGER,
            text_digits REAL,
            text_uppercase REAL,
            text_sentiment REAL,
            text_max_cosine_similarity REAL,
            text_different_tokens INTEGER,
            text_description_similarity REAL,
            hotel_mention INTEGER,
            score_deviation REAL
        )""")


def create_reviewers_table():
    with conn:
        c.execute("""CREATE TABLE IF NOT EXISTS reviewers (
            username_id TEXT PRIMARY KEY,
            user_location TEXT,
            user_register_date TEXT,
            number_of_reviews INTEGER,
            maximum_reviews INTEGER,
            helpful_vote INTEGER,
            number_of_good_reviews REAL,
            number_of_bad_reviews REAL,
            average_score REAL,
            median_score REAL,
            average_text_characters REAL,
            average_text_sentences REAL,
            max_similarity REAL,
            first_review_date TEXT,
            last_review_date TEXT,
            max_reviews_on_same_hotel INTEGER
        )""")


def insert_hotel(hotel: dict):
    with conn:
        c.execute(
            """INSERT INTO hotels VALUES (:hotel_id, :average_score, :median_score, :number_of_reviews,
            :deviation, :max_review_one_day, :distortion, :good_rating_one_day, :bad_rating_one_day)""",
            {'hotel_id': hotel['hotel_id'],
             'average_score': hotel['average_score'],
             'median_score': hotel['median_score'],
             'number_of_reviews': hotel['number_of_reviews'],
             'deviation': hotel['deviation'],
             'max_review_one_day': hotel['max_review_one_day'],
             'distortion': hotel['distortion'],
             'good_rating_one_day': hotel['good_rating_one_day'],
             'bad_rating_one_day': hotel['bad_rating_one_day']
             })


def insert_review(review: dict):
    with conn:
        c.execute(
            """INSERT INTO reviews VALUES (
            :review_id, :username_id, :hotel_id, :review_date, :date_of_stay, :score, :title, :text, :title_length,
            :text_length, :text_sentences, :text_digits, :text_uppercase, :text_sentiment, :text_max_cosine_similarity,
            :text_different_tokens, :text_description_similarity, :hotel_mention, :score_deviation)""",
            {
                'review_id': review['review_id'],
                'username_id': review['username_id'],
                'hotel_id': review['hotel_id'],
                'review_date': review['review_date'],
                'date_of_stay': review['date_of_stay'],
                'score': review['score'],
                'title': review['title'],
                'text': review['text'],
                'title_length': review['title_length'],
                'text_length': review['text_length'],
                'text_sentences': review['text_sentences'],
                'text_digits': review['text_digits'],
                'text_uppercase': review['text_uppercase'],
                'text_sentiment': review['text_sentiment'],
                'text_max_cosine_similarity': review['text_max_cosine_similarity'],
                'text_different_tokens': review['text_different_tokens'],
                'text_description_similarity': review['text_description_similarity'],
                'hotel_mention': review['hotel_mention'],
                'score_deviation': review['score_deviation']
            })


def insert_reviewer(reviewer: dict):
    with conn:
        c.execute(
            """INSERT INTO reviewers VALUES (
            :username_id, :user_location, :user_register_date, :number_of_reviews, :maximum_reviews, :helpful_vote,
            :number_of_good_reviews, :number_of_bad_reviews, :average_score, :median_score, :average_text_characters,
            :average_text_sentences, :max_similarity, :first_review_date, :last_review_date, :max_reviews_on_same_hotel)""",
            {'username_id': reviewer['username_id'],
             'user_location': reviewer['user_location'],
             'user_register_date': reviewer['user_register_date'],
             'number_of_reviews': reviewer['number_of_reviews'],
             'maximum_reviews': reviewer['maximum_reviews'],
             'helpful_vote': reviewer['helpful_vote'],
             'number_of_good_reviews': reviewer['number_of_good_reviews'],
             'number_of_bad_reviews': reviewer['number_of_bad_reviews'],
             'average_score': reviewer['average_score'],
             'median_score': reviewer['median_score'],
             'average_text_characters': reviewer['average_text_characters'],
             'average_text_sentences': reviewer['average_text_sentences'],
             'max_similarity': reviewer['max_similarity'],
             'first_review_date': reviewer['first_review_date'],
             'last_review_date': reviewer['last_review_date'],
             'max_reviews_on_same_hotel': reviewer['max_reviews_on_same_hotel']
             })


def get_hotels():
    c.execute("SELECT * FROM hotels")
    return c.fetchall()
