import json
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


def get_sum_of_reviews():
    pass


def get_sum_of_same_day_reviews():
    pass


def main():
    data = load_json('tripadvisor.json')

    text = """This my the 3 rd time in this hotel. The service is very NICE.
                                They know me and know my tastes.
                                I love the breakfast and Bruno are fantastic and so efficient by delivering the best breakfast I can see in this area.
                                They serve you with a smile and this is so nice !"""
    print(get_number_of_characters(text))
    print(get_number_of_sentences(text))
    print(get_number_of_different_token(text))
    print(get_percentage_of_digit(text))
    print(get_percentage_of_uppercase_words(text))
    print(get_number_of_hotel_name_mention('I', text))
    print(get_deviation_from_rating(4.5, 2))
    print(get_deviation_from_rating(2, 5))
    print(stopwords)
    print(remove_stopwords('Das ist ein Test'))
    print(get_cosine_similarity('Das ist ein komischer Test heute von Frau Dernd',
                                'Was wollen wir heute machen an diesen schönen Tag?'))


if __name__ == '__main__':
    stopwords = stopwords.words('german')
    main()
