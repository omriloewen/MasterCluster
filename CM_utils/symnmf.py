""" Python interface """

from . import mysymnmf
import numpy as np
import pandas as pd
import math


def input_to_X(input_file):
    """
    Reads input txt file into matrix X

    Args:
        input_file (str): The path of the file (from root of this file)

    Returns:
            tuple of:
        X (list of list of float): The data in the file
        N (int): Number of datapoints in input file
        d (int): Dimention of datapoints
    """
    readfile = pd.read_csv(input_file, header=None, delimiter=",")
    X = readfile.to_numpy().tolist()
    return X


def gen_H(W, k):
    """
    Generates the initial H matrix for the symnmf algorithem

    Args:
        W (numpy ndarray): The normalized similarity matrix
        k (int): The number of requested clusters

    Returns:
        H0 (numpy ndarray): Initialized H0 matrix
    """
    # initialize H
    m = np.mean(np.asarray(W)).item()
    return (np.random.uniform(0, 2 * math.sqrt(m / k), (len(W), k))).tolist()


def symnmf_labels(W, H_init):
    H = mysymnmf.symnmf(W, H_init, len(W), len(H_init[0]))
    labels = [H[i].index(max(H[i])) for i in range(len(H))]
    return labels


def symnmf(filepath, k):
    X = input_to_X(filepath)
    W = mysymnmf.norm(X, len(X), len(X[0]))
    H = gen_H(W, k)
    return symnmf_labels(W, H)
