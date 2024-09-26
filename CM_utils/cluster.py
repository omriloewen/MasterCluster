from . import kmeanspp
from . import symnmf
import pandas as pd
from sklearn.metrics import silhouette_score


def cluster(filepath, k):
    kmeans_labels = kmeanspp.kmeanspp(filepath, k)
    symnmf_labels = symnmf.symnmf(filepath, k)
    X = pd.read_csv(filepath, header=None, delimiter=",")
    Kmeans_score = silhouette_score(X, kmeans_labels)
    symnmf_score = silhouette_score(X, symnmf_labels)
    if Kmeans_score > symnmf_score:
        print("chose kmeans with score: ", Kmeans_score)
        return kmeans_labels

    else:
        print("chose symnmf with score: ", symnmf_score)
        return symnmf_labels
