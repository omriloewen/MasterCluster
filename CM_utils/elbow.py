from . import kmeanspp
from . import symnmf
import pandas as pd
from sklearn.metrics import silhouette_score


def cluster(filepath):
    kmeans_res = elbow_kmeans(filepath)
    symnmf_res = elbow_symnmf(filepath)
    if kmeans_res[1] > symnmf_res[1]:
        print("chose kmeans with score: ", kmeans_res[1])
        return kmeans_res[0]
    else:
        print("chose symnmf with score: ", symnmf_res[1])
        return symnmf_res[0]


def elbow_kmeans(filepath):
    maxk = 50
    X = pd.read_csv(filepath, header=None, delimiter=",")
    labels2 = kmeanspp.kmeanspp(filepath, 2)
    score_m3 = silhouette_score(X, labels2)
    labels_m2 = kmeanspp.kmeanspp(filepath, 3)
    score_m2 = silhouette_score(X, labels_m2)
    d_m2 = score_m2 - score_m3
    if d_m2 < 0.02:
        return labels2, score_m3
    labels_m1 = kmeanspp.kmeanspp(filepath, 4)
    score_m1 = silhouette_score(X, labels_m1)
    d_m1 = score_m1 - score_m2
    for k in range(5, min(len(X), maxk)):
        labels = kmeanspp.kmeanspp(filepath, k)
        score = silhouette_score(X, labels)
        d = score - score_m1
        if d < d_m1 and d_m1 < d_m2:
            return labels_m2, score_m2
        else:
            labels_m2 = labels_m1.copy()
            labels_m1 = labels.copy()
            score_m2 = score_m1
            score_m1 = score
            d_m2 = d_m1
            d_m1 = d

    return labels, score


def elbow_symnmf(filepath):
    maxk = 50
    X = pd.read_csv(filepath, header=None, delimiter=",")
    labels2 = symnmf.symnmf(filepath, 2)
    score_m3 = silhouette_score(X, labels2)
    labels_m2 = symnmf.symnmf(filepath, 3)
    score_m2 = silhouette_score(X, labels_m2)
    d_m2 = score_m2 - score_m3
    if d_m2 < 0.02:
        return labels2, score_m3
    labels_m1 = symnmf.symnmf(filepath, 4)
    score_m1 = silhouette_score(X, labels_m1)
    d_m1 = score_m1 - score_m2
    for k in range(5, min(len(X), maxk)):
        labels = symnmf.symnmf(filepath, k)
        score = silhouette_score(X, labels)
        d = score - score_m1
        if d < d_m1 and d_m1 < d_m2:
            return labels_m2, score_m2
        else:
            labels_m2 = labels_m1.copy()
            labels_m1 = labels.copy()
            score_m2 = score_m1
            score_m1 = score
            d_m2 = d_m1
            d_m1 = d

    return labels, score
