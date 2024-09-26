import numpy as np
import pandas as pd
from . import mykmeanssp


# get the data point from an input file
def read_file(path):
    readFile = pd.read_csv(path, header=None, delimiter=",")
    X = readFile.to_numpy().tolist()
    return X


def init_centroids(vectors, k):  # get the vectors from the input files
    N = len(vectors)

    if k < 2 or k >= N:  # assert valid number of clusters
        return "Invalid number of clusters"

    D = np.zeros(N)  # min distances as defined
    cents = []  # this wil hold the initial centers chosen

    # choose the first center uniformly
    cent_ind = np.random.choice(N)
    chosen_cent = vectors[cent_ind]
    cents.append(chosen_cent.copy())

    for i in range(1, k):  # until k centers were chosen

        for j in range(N):  # compute each vector min distance
            D[j] = min(
                np.linalg.norm(np.array(vectors[j]) - np.array(cent)) for cent in cents
            )

        probs = D / np.sum(D)  # set the new probabilities

        # choose the next center by the new probabilities
        cent_ind = np.random.choice(N, p=probs)
        chosen_cent = vectors[cent_ind]
        cents.append(chosen_cent.copy())

    return cents


def kmeans_labels(vectors, cents, k, iter=300, e=0.001):
    N = len(vectors)
    d = len(vectors[0])
    labels = mykmeanssp.fit(vectors, cents, N, d, k, iter, e)
    return labels


def kmeanspp(file_path, k):
    X = read_file(file_path)
    centroids = init_centroids(X, k)
    return kmeans_labels(X, centroids, k)
