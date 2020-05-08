import pandas as pd
import numpy as np
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# nltk.download('punkt')


# START #
def load_data(path):
    """
    :param path:
    :return: data, label
    """
    dataset = np.array(pd.read_csv(path, header=None, skiprows=1))
    data = dataset[:, 2]
    label = dataset[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.5)
    return data, label, x_train, x_test, y_train, y_test


def get_word_list(data, size):
    """
    :param data:
    :param size:
    :return: the biggest rank by size
    """
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()

    word_list = []

    for item in data:
        item = re.sub(r'[^a-zA-Z0-9\s]', '', item)
        word_token = word_tokenize(item)
        for word in word_token:
            word = porter_stemmer.stem(word)
            if word not in stop_words:
                word_list.append(word)

    word_list = [[k[0] for k in Counter(word_list).most_common(size)]]

    return word_list


def prepare_data(word_list, size, x_train, y_train, seq_length):
    """
    :param word_list:
    :param size:
    :param x_train:
    :param y_train:
    :param seq_length:
    :return:
    """
    if len(word_list) < size:
        size = len(word_list)
    content_length_list = [len(k) for k in x_train]
    if max(content_length_list) < seq_length:
        seq_length = max(content_length_list)
    word_id_list = dict([(b, a) for a, b in enumerate(word_list)])
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    return label_encoder, word_id_list, seq_length, size


def save(x_train, x_test, y_train, y_test, data, label, word_list, label_encoder, word_id_list, seq_length, size):
    content = {"x_train": x_train,
               "x_test": x_test,
               "y_train": y_train,
               "y_test": y_test,
               "data": data,
               "word_list": word_list,
               "label": label,
               "label_encoder": label_encoder,
               "word_id_list": word_id_list,
               "seq_length": seq_length,
               "size": size}
    # print(content)
    with open("../data/data.pickle", "wb") as f:
        pickle.dump(content, f)
        print("Deal file finished...")


def main():
    path = "../data/spam_ham_dataset.csv"
    size = 1000
    seq_length = 230
    data, label, x_train, x_test, y_train, y_test = load_data(path)
    word_list = get_word_list(data, size)
    label_encoder, word_id_list, seq_length, size = prepare_data(word_list[0], size, x_train, y_train, seq_length)
    save(x_train, x_test, y_train, y_test, data, label, word_list, label_encoder, word_id_list, seq_length, size)


if __name__ == '__main__':
    main()
# END #
