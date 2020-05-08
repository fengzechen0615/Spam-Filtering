import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def load_data(path):
    dataset = np.array(pd.read_csv(path, header=None, skiprows=1))
    label = dataset[:, -1]
    data = dataset[:, 2]
    length = 0
    count = 0
    for item in data:
        length += len(item.split())
        count += 1
    # print(length/count)
    # print(Counter(label))
    # plot(Counter(label))
    deal_word(data)


def deal_word(data):
    word_list = []
    stop_words = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    for item in data:
        item = re.sub(r'[^a-zA-Z0-9\s]', '', item)
        word_token = word_tokenize(item)
        for word in word_token:
            word = porter_stemmer.stem(word)
            if word not in stop_words:
                word_list.append(word)
    print(word_list)
    print(len(Counter(word_list).keys()))
    print(Counter(word_list))
    # word_list = [[k[0] for k in Counter(word_list).most_common(size)]?


def plot(data):
    name_list = ["Spam", "Ham"]
    name_list = [0, 1]
    value_list = [data.get(1), data.get(0)]
    for a, b in zip(name_list, value_list):
        plt.text(a, b + 0.1, '%.0f' % b, ha='center', va='bottom', fontsize=14)
    plt.bar(range(len(value_list)), value_list, width=0.75, tick_label=name_list)
    plt.show()


def main():
    path = "../data/spam_ham_dataset.csv"
    load_data(path)


if __name__ == '__main__':
    main()
