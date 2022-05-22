import unittest
import main

import pandas as pd
import numpy as np


class TestMain(unittest.TestCase):
    def test_get_reviews_by_hotel_id(self):
        df = pd.DataFrame({'hotel_id': ['42', '33', '32', '42'],
                           'review_id': ['1', '2', '3', '4'],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        df_expected = pd.DataFrame({'hotel_id': ['42', '42'],
                                    'review_id': ['1', '4'],
                                    'review_text': ['The hotel Hilton was nice',
                                                    'The best Hilton in Switzerland']})
        df_res = main.get_reviews_by_hotel_id(df, '42')
        np.array_equal(df_res.values, df_expected.values)

    def test_get_reviews_by_username_id(self):
        df = pd.DataFrame({'hotel_id': ['42', '33', '32', '42'],
                           'username_id': ['3', '2', '3', '4'],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        df_expected = pd.DataFrame({'hotel_id': ['42', '32'],
                                    'username_id': ['3', '3'],
                                    'review_text': ['The hotel Hilton was nice',
                                                    'I did not liked the Hilton hotel']})
        df_res = main.get_reviews_by_username_id(df, '3')
        np.array_equal(df_res.values, df_expected.values)

    def test_get_max_cosine_similarity_hotel(self):
        df = pd.DataFrame({'hotel_id': ['42', '42', '42', '42'],
                           'review_id': [1, 2, 3, 4],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        res = main.get_max_cosine_similarity_hotel(df, '42', '1', 'The hotel Hilton was nice')
        self.assertAlmostEqual(res, 0.8944)

    # How to test this? Get Error NameError: name 'sentiment_model' is not defined
    # def test_get_sentiment(self):
    #     res = main.get_sentiment('Das Hotel war super schön. Hat mir sehr gefallen.')
    #     self.assertGreaterEqual(res, 0.9)
    #
    #     res = main.get_sentiment('Das Hotel war sehr schlecht. Die Zimmer waren dreckig.')
    #     self.assertLessEqual(res, -0.9)

    # Reviewer specific
    def test_get_number_of_reviews_for_reviewer(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        res = main.get_number_of_reviews_for_reviewer(df)
        self.assertEqual(res, 4)

    def test_get_max_reviews_on_one_day(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_date': ['2020-02-10', '2020-10-02', '2022-03-10', '2020-02-10']})
        res = main.get_max_reviews_on_one_day(df)
        self.assertEqual(res, 2)

    def test_get_sum_of_helpful_votes(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_helpful_vote': [3, 0, 2, 1]})
        res = main.get_sum_of_helpful_votes(df)
        self.assertEqual(res, 6)

    def test_get_number_of_good_rating(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_score': [4, 5, 3, 2]})
        res = main.get_number_of_good_rating(df)
        self.assertEqual(res, 0.5)

        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_score': [3, 1, 3, 2]})
        res = main.get_number_of_good_rating(df)
        self.assertEqual(res, 0)

    def test_get_number_of_bad_rating(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_score': [4, 5, 3, 2]})
        res = main.get_number_of_bad_rating(df)
        self.assertEqual(res, 0.25)

        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_score': [3, 5, 3, 4]})
        res = main.get_number_of_bad_rating(df)
        self.assertEqual(res, 0)

    def test_get_average_score(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_score': [4, 5, 3, 2]})
        res = main.get_average_score(df)
        self.assertEqual(res, 3.5)

    def test_get_median_score_reviewer(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_score': [4, 5, 3, 2]})
        res = main.get_median_score_reviewer(df)
        self.assertEqual(res, 3.5)

    def test_get_average_text_characters(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        res = main.get_average_text_characters(df)
        self.assertEqual(res, 26.25)

    def test_get_average_text_sentences(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_text': ['The hotel Hilton was nice. Just beautiful',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel. It was terrible! Not recommended',
                                           'The best Hilton in Switzerland']})
        res = main.get_average_text_sentences(df)
        self.assertEqual(res, 1.75)

    def test_get_date_of_first_review(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_date': ['2020-03-10', '2020-10-02', '2022-03-10', '2020-02-10']})
        res = main.get_date_of_first_review(df)
        self.assertEqual(res, '2020-02-10')

    def test_get_date_of_last_review(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_date': ['2020-03-10', '2020-10-02', '2022-03-10', '2020-02-10']})
        res = main.get_date_of_last_review(df)
        self.assertEqual(res, '2022-03-10')

    def test_get_max_reviews_on_same_hotel(self):
        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'hotel_id': [1, 2, 3, 4],
                           'review_text': ['The hotel Hilton was nice. Just beautiful',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel. It was terrible! Not recommended',
                                           'The best Hilton in Switzerland']})
        res = main.get_max_reviews_on_same_hotel(df)
        self.assertEqual(res, 1)

        df = pd.DataFrame({'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'hotel_id': [1, 4, 3, 4],
                           'review_text': ['The hotel Hilton was nice. Just beautiful',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel. It was terrible! Not recommended',
                                           'The best Hilton in Switzerland']})
        res = main.get_max_reviews_on_same_hotel(df)
        self.assertEqual(res, 2)

    def test_get_max_cosine_similarity_reviewer(self):
        df = pd.DataFrame({'hotel_id': ['42', '43', '44', '55'],
                           'username_id': ['aley', 'aley', 'aley', 'aley'],
                           'review_id': [1, 2, 3, 4],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        res = main.get_max_cosine_similarity_reviewer(df)
        self.assertAlmostEqual(res, 0.8944)

    # Hotel specific
    def test_get_median_of_hotel_score(self):
        df = pd.DataFrame({'hotel_id': ['42', '33', '32', '42', '42', '33', '42', '33'],
                           'review_score': [3, 2, 3, 4, 5, 3, 2, 1]})
        res = main.get_median_of_hotel_score(df, '42')
        self.assertEqual(res, 3.5)

    def test_get_number_of_reviews_by_hotel_id(self):
        df = pd.DataFrame({'hotel_id': ['42', '33', '32', '42'],
                           'review_id': ['1', '2', '3', '4'],
                           'review_text': ['The hotel Hilton was nice',
                                           'The hotel was nice',
                                           'I did not liked the Hilton hotel',
                                           'The best Hilton in Switzerland']})
        res = main.get_number_of_reviews_by_hotel_id(df, '42')
        self.assertEqual(res, 2)

        res = main.get_number_of_reviews_by_hotel_id(df, '52')
        self.assertEqual(res, 0)

    def test_get_hotel_score_deviation(self):
        df = pd.DataFrame({'hotel_id': ['42', '33', '32', '42', '42', '33', '42', '33'],
                           'review_score': [3, 2, 3, 4, 5, 3, 2, 1]})
        res = main.get_hotel_score_deviation(df, '42')
        expected = np.std([3, 4, 5, 2])
        self.assertEqual(res, expected)

        res = main.get_hotel_score_deviation(df, '33')
        expected = np.std([2, 3, 1])
        self.assertEqual(res, expected)

    def test_get_max_review_percentage_on_one_day(self):
        df = pd.DataFrame({'hotel_id': ['42', '42', '32', '42', '42'],
                           'review_id': [1, 2, 3, 4, 5],
                           'review_date': ['2020-03-10', '2020-03-02', '2022-03-10', '2020-03-10', '2020-03-11']})
        res = main.get_max_review_percentage_on_one_day(df, '42', 4)
        self.assertEqual(res, 0.5)

    def test_get_number_good_rating_on_one_day(self):
        df = pd.DataFrame({'hotel_id': ['42', '42', '32', '42', '42', '33'],
                           'review_score': [5, 2, 5, 4, 5, 2],
                           'review_date': ['2020-03-10', '2020-03-02', '2020-03-10', '2020-03-10', '2020-03-11',
                                           '2020-03-11']})
        res = main.get_number_good_rating_on_one_day(df, '42', 4)
        self.assertEqual(res, 0.5)

        res = main.get_number_good_rating_on_one_day(df, '32', 1)
        self.assertEqual(res, 1)

        res = main.get_number_good_rating_on_one_day(df, '33', 1)
        self.assertEqual(res, 0)

        res = main.get_number_good_rating_on_one_day(df, '23', 0)
        self.assertEqual(res, 0)

    def test_get_number_bad_rating_on_one_day(self):
        df = pd.DataFrame({'hotel_id': ['42', '42', '32', '42', '42', '33'],
                           'review_score': [2, 2, 1, 4, 2, 5],
                           'review_date': ['2020-03-10', '2020-03-02', '2020-03-10', '2020-03-10', '2020-03-11',
                                           '2020-03-11']})
        res = main.get_number_bad_rating_on_one_day(df, '42', 4)
        self.assertEqual(res, 0.25)

        res = main.get_number_bad_rating_on_one_day(df, '32', 1)
        self.assertEqual(res, 1)

        res = main.get_number_bad_rating_on_one_day(df, '33', 1)
        self.assertEqual(res, 0)

        res = main.get_number_bad_rating_on_one_day(df, '23', 0)
        self.assertEqual(res, 0)

    def test_get_hotel_score_distortion(self):
        df = pd.DataFrame(
            {'hotel_id': ['42', '42', '22', '22', '42', '42', '42', '42', '42', '42', '42', '42', '42', '42'],
             'review_score': [4, 5, 5, 4, 5, 4, 5, 5, 5, 4, 4, 4, 5, 4],
             'hotel_score': [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]})
        res = main.get_hotel_score_distortion(df, '42')
        self.assertLessEqual(res, 0.2)

        df = pd.DataFrame(
            {'hotel_id': ['42', '42', '22', '22', '42', '42', '42', '42', '42', '42', '42', '42', '42', '42'],
             'review_score': [2, 5, 5, 1, 5, 3, 5, 4, 5, 2, 3, 3, 5, 1],
             'hotel_score': [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]})
        res = main.get_hotel_score_distortion(df, '42')
        self.assertGreaterEqual(res, 0.4)


if __name__ == '__main__':
    unittest.main()
