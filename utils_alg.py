import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
import scipy.linalg as linalg
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import sys
from os import path
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import random


def absorption_probability(W, alpha, stored_A=None, column=None):
    try:
        # raise Exception('DEBUG')
        A = np.load(stored_A + str(alpha) + '.npz')['arr_0']
        print('load A from ' + stored_A + str(alpha) + '.npz')
        if column is not None:
            P = np.zeros(W.shape)
            P[:, column] = A[:, column]
            return P
        else:
            return A
    except:
        # W=sp.csr_matrix([[0,1],[1,0]])
        # alpha = 1
        n = W.shape[0]
        print('Calculate absorption probability...')
        W = W.copy().astype(np.float32)
        D = W.sum(1).flat
        L = sp.diags(D, dtype=np.float32) - W
        L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
        L = sp.csc_matrix(L)
        # print(np.linalg.det(L))

        if column is not None:
            A = np.zeros(W.shape)
            # start = time.time()
            A[:, column] = slinalg.spsolve(L, sp.csc_matrix(np.eye(L.shape[0], dtype='float32')[:, column])).toarray()
            # print(time.time()-start)
            return A
        else:
            # start = time.time()
            A = slinalg.inv(L).toarray()
            # print(time.time()-start)
            if stored_A:
                np.savez(stored_A + str(alpha) + '.npz', A)
            return A
            # fletcher_reeves

            # slinalg.solve(L, np.ones(L.shape[0]))
            # A_ = np.zeros(W.shape)
            # I = sp.eye(n)
            # Di = sp.diags(np.divide(1,np.array(D)+alpha))
            # for i in range(10):
            #     # A_=
            #     A_ = Di*(I+W.dot(A_))
            # print(time.time()-start)


def cotraining(W, t, alpha, y_train, train_mask, stored_A=None):
    A = absorption_probability(W, alpha, stored_A, train_mask)
    y_train = y_train.copy()
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    # if not isinstance(features, np.ndarray):
    #     features = features.toarray()
    print("Additional Label:")
    if not hasattr(t, '__getitem__'):
        t = [t for _ in range(y_train.shape[1])]
    for i in range(y_train.shape[1]):
        y = y_train[:, i:i + 1]
        a = A.dot(y)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]

        # x1 = features[index, :].reshape((-1, 1, features.shape[1]))
        # x2 = features[y_train[:, i].astype(np.bool)].reshape((1, -1, features.shape[1]))
        # D = np.sum((x1 - x2) ** 2, axis=2) ** 0.5
        # D = np.mean(D, axis=1)
        # gate = 100000000 if t[i] >= D.shape[0] else np.sort(D, axis=0)[t[i]]
        # index = index[D<gate]
        train_index = np.hstack([train_index, index])
        y_train[index, i] = 1
        correct_label_count(index, i)
    print()
    train_mask = sample_mask(train_index, y_train.shape[0])
    return y_train, train_mask


def selftraining(prediction, t, y_train, train_mask):
    new_gcn_index = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    no_class = y_train.shape[1]  # number of class
    if hasattr(t, '__getitem__'):
        assert len(t) >= no_class
        index = []
        count = [0 for i in range(no_class)]
        for i in sorted_index:
            for j in range(no_class):
                if new_gcn_index[i] == j and count[j] < t[j] and not train_mask[i]:
                    index.append(i)
                    count[j] += 1
    else:
        index = sorted_index[:t]

    indicator = np.zeros(train_mask.shape, dtype=np.bool)
    indicator[index] = True
    indicator = np.logical_and(np.logical_not(train_mask), indicator)

    prediction = np.zeros(prediction.shape)
    prediction[np.arange(len(new_gcn_index)), new_gcn_index] = 1.0
    prediction[train_mask] = y_train[train_mask]

    correct_labels = np.sum(prediction[indicator] * all_labels[indicator], axis=0)
    count = np.sum(prediction[indicator], axis=0)
    print('Additiona Label:')
    for i, j in zip(correct_labels, count):
        print(int(i), '/', int(j), sep='', end='\t')
    print()

    y_train = np.copy(y_train)
    train_mask = np.copy(train_mask)
    train_mask[indicator] = 1
    y_train[indicator] = prediction[indicator]
    return y_train, train_mask


def lp(adj, alpha, y_train, train_mask, y_test, stored_A=None):
    P = absorption_probability(adj, alpha, stored_A=stored_A, column=train_mask)
    P = P[:, train_mask]

    # nearest clssifier
    predicted_labels = np.argmax(P, axis=1)
    # prediction = alpha*P
    prediction = np.zeros(P.shape)
    prediction[np.arange(P.shape[0]), predicted_labels] = 1

    y = np.sum(train_mask)
    label_per_sample = np.vstack([np.zeros(y), np.eye(y)])[np.add.accumulate(train_mask) * train_mask]
    sample2label = label_per_sample.T.dot(y_train)
    prediction = prediction.dot(sample2label)

    test_acc = np.sum(prediction * y_test) / np.sum(y_test)
    test_acc_of_class = np.sum(prediction * y_test, axis=0) / np.sum(y_test, axis=0)
    # print(test_acc, test_acc_of_class)
    return test_acc, test_acc_of_class, prediction


def union_intersection(prediction, t, y_train, train_mask, W, alpha, stored_A, union_or_intersection):
    no_class = y_train.shape[1]  # number of class

    # gcn index
    new_labels_gcn = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    sorted_index = np.argsort(-confidence)

    if not hasattr(t, '__getitem__'):
        t = [t for i in range(no_class)]

    assert len(t) >= no_class
    count = [0 for i in range(no_class)]
    index_gcn = [[] for i in range(no_class)]
    for i in sorted_index:
        j = new_labels_gcn[i]
        if count[j] < t[j] and not train_mask[i]:
            index_gcn[j].append(i)
            count[j] += 1

    # lp
    A = absorption_probability(W, alpha, stored_A, train_mask)
    train_index = np.where(train_mask)[0]
    already_labeled = np.sum(y_train, axis=1)
    index_lp = []
    for i in range(no_class):
        y = y_train[:, i:i + 1]
        a = np.sum(A[:, y.flat > 0], axis=1)
        a[already_labeled > 0] = 0
        # a[W.dot(y) > 0] = 0
        gate = (-np.sort(-a, axis=0))[t[i]]
        index = np.where(a.flat > gate)[0]
        index_lp.append(index)

    # print(list(map(len, index_gcn)))
    # print(list(map(len, index_lp)))

    y_train = y_train.copy()
    print("Additional Label:")
    for i in range(no_class):
        assert union_or_intersection in ['union', 'intersection']
        if union_or_intersection == 'union':
            index = list(set(index_gcn[i]) | set(index_lp[i]))
        else:
            index = list(set(index_gcn[i]) & set(index_lp[i]))
        y_train[index, i] = 1
        train_mask[index] = True
        print(np.sum(all_labels[index, i]), '/', len(index), sep='', end='\t')
    return y_train, train_mask


def ap_approximate(adj, features, alpha, k):
    adj = normalize(adj + sp.eye(adj.shape[0]), 'l1', axis=1) / (alpha + 1)
    # D = sp.diags(np.array(adj.sum(axis=1)).flatten())+alpha*sp.eye(adj.shape[0])
    # D = D.power(-1)
    # adj = D*adj
    # features = D*alpha*features
    if sp.issparse(features):
        features = features.toarray()
    new_feature = np.zeros(features.shape)
    for _ in range(k):
        new_feature = adj * new_feature + features
    new_feature *= alpha / (alpha + 1)
    return new_feature


all_labels = None


# dataset = None

def correct_label_count(indicator, i):
    count = np.sum(all_labels[:, i][indicator])
    if indicator.dtype == np.bool:
        total = np.where(indicator)[0].shape[0]
    elif indicator.dtype in [np.int, np.int8, np.int16, np.int32, np.int64]:
        total = indicator.shape[0]
    else:
        raise TypeError('indicator must be of data type np.bool or np.int')
    # print("     for class {}, {}/{} is correct".format(i, count, total))
    print(count, '/', total, sep='', end='\t')


