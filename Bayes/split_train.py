#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project -> File   ：Spam-Filtering -> split_train
@IDE    ：PyCharm
@Author ：Yilin Lou
@Date   ：5/5/20 9:49 下午
@Group  ：Stevens Institute of technology
'''
import numpy as np


def split_train(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train = data.iloc[train_indices]
    test = data.iloc[test_indices]
    return train, test