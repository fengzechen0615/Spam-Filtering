#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：Spam-Filtering -> bayes
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：5/5/20 7:27 下午
@Group  ：Stevens Institute of technology
'''

import time
import nltk
import ssl
import time
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))     #load stopwords
# print(STOPWORDS)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD  # 降维
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB  # 伯努利分布的贝叶斯公式
from sklearn.metrics import f1_score, precision_score, recall_score
#############################################################################   start
# 1、文件数据读取
from Bayes.split_train import split_train

df = pd.read_csv("../Data/spam_ham_dataset.csv", encoding="utf-8", sep=",")

# Removing Unnecessary column
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(axis=0, how="any", inplace=True)  # delete NA
# print(df.head())
# print(df.info())
# Changing column names
df.columns = ['label', 'text', 'class']
# print(df.shape)

df.drop('label', axis=1, inplace=True)
# print(df.shape)

# for picture
# accuracy list
acc=[]

# time list
times=[]
# PCV
components=np.arange(10,510,10)
a=np.arange(1,11,1)

#components=[10,20,30,40,50,60,70,80,90,100]

#
for m in range(10):
    start = time.time()
    # 2、train and test
    # x_train, x_test, y_train, y_test = train_test_split(df, df["class"], test_size=0.2, random_state=0)
    train, test = split_train(df, 0.5)
    x_train = train['text']
    x_test = test['text']
    y_train = train['class']
    y_test = test['class']

    # print("number of train :%d" % x_train.shape[0])
    # print("number if test :%d" % x_test.shape[0])
    # print(x_train.head())

    # change text to digital number
    '''
    TF-IDF(term frequency-inverse document frequency)词频-逆向文件频率。在处理文本时，如何将文字转化为模型可以处理的向量呢？TF-IDF就是这个问题的解决方案之一。字词的重要性与其在文本中出现的频率成正比(TF)，与其在语料库中出现的频率成反比(IDF)。

    TF
    TF：词频。TF(w)=(词w在文档中出现的次数)/(文档的总词数)

    IDF
    IDF：逆向文件频率。有些词可能在文本中频繁出现，但并不重要，也即信息量小，如is,of,that这些单词，这些单词在语料库中出现的频率也非常大，我们就可以利用这点，降低其权重。IDF(w)=log_e(语料库的总文档数)/(语料库中词w出现的文档数)

    TF-IDF
    将上面的TF-IDF相乘就得到了综合参数：TF-IDF=TF*IDF
    '''

    '''
    norm='l2'范数时，就是对文本向量进行归一化
    use_idf：boolean， optional
        启动inverse-document-frequency重新计算权重
    Stopwords
    '''
    # stop_words = set(stopwords.words('english'))
    # stop = list(stopwords.words('english'))

    # print(stop)
    transformer = TfidfVectorizer(norm="l2", use_idf=True,
                                  stop_words=STOPWORDS)  # TF-IDF(term frequency-inverse document frequency)词频-逆向文件频率
    svd = TruncatedSVD(n_components=200)  # 奇异值分解，降维
    # print(svd)
    train_cut_text = list(x_train.astype("str"))
    # print(train_cut_text)
    transformer_model = transformer.fit(train_cut_text)
    # print(transformer_model)

    df1 = transformer_model.transform(train_cut_text)

    svd_model = svd.fit(df1)
    df2 = svd_model.transform(df1)  # 降维
    # print(df2)
    data = pd.DataFrame(df2)

    # print(data.info())

    # data["class"] = list(y_train)

    #################################################

    nb1 = BernoulliNB(alpha=1.0, binarize=0.001)  # Bayes
    nb2 = GaussianNB()  # GaussianNB

    model = nb2.fit(data, y_train)

    # print(nb1.score(data, y_train))

    # change text to sperated str
    cut_test = list(x_test.astype("str"))
    data_test = pd.DataFrame(svd_model.transform(transformer_model.transform(cut_test)))

    # data_test["class"] = list(y_test)
    # print(data_test.head())
    # print(data_test.info())

    #  predict
    y_predict = model.predict(data_test)

    # Accuracy rate

    y_test = np.array(y_test)
    y_predict = np.array(y_predict)

    count = 0
    fail = 0
    for i in range(0, len(y_test)):
        if (y_test[i] == y_predict[i]):
            count = count + 1
        else:
            fail = fail + 1
    k=m+1
    print("The %d time"%k)
    accuracy=count / len(y_test)

    print("Accuracy rate: ", accuracy)

    print("The times uncorrect are : ", fail)

    end = (time.time() - start)
    print("Time used:", end)

    print('\n')
    acc.append(accuracy)
    times.append(end)


print("The average accyracy is : ",np.mean(acc))

# plot函数作图
# plt.plot(components,acc)
# plt.savefig("components_acc.png")
#
# # show函数展示出这个图，如果没有这行代码，则程序完成绘图，但看不到
# plt.show()
#
# plt.plot(components,times)
# plt.savefig("components_times.png")
#
# plt.show()

plt.plot(a,acc)
plt.title('GaussianNB--Accuracy: 10 times')
plt.savefig("GaussianNB_acc.png")
plt.show()

# plt.plot(a,acc)
# plt.title('BernoulliNB--Accuracy: 10 times')
# plt.savefig("BernoulliNB_acc.png")
# plt.show()

########################################################################   end