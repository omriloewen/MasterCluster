import numpy as np
import pandas as pd
import mysymnmf


def input_to_X(input_file):
    """Reads a CSV file and converts it into a list of lists.

    Args:
        input_file (str): The path to the CSV file to be read.

    Returns:
        list: A list of lists containing the data from the CSV file.
    """
    print("symnmf.input_to_X")
    # Use pandas to read the CSV file without headers and convert to a list of lists
    return pd.read_csv(input_file).values.tolist()


def gen_H(W, k):
    """Generates an initial matrix H using a uniform distribution.

    Args:
        W (numpy.ndarray): The input matrix W used in the NMF process.
        k (int): The number of components for the matrix H.

    Returns:
        list: A list representing the generated matrix H.
    """
    print(f"generating initial H matrix for k = {k}")
    # Calculate the mean of the elements in W
    mean_W = np.mean(W)
    # Generate a uniform distribution for H based on the mean and number of components k
    return np.random.uniform(0, 2 * np.sqrt(mean_W / k), (len(W), k)).tolist()


def symnmf_labels(W, H_init):
    """Generates labels based on the maximum value in each row of matrix H after applying NMF.

    Args:
        W (array): The input matrix W used in NMF.
        H_init (list): The initial guess for matrix H.

    Returns:
        list: A list of labels, where each label corresponds to the index of the maximum element in each row of H.
    """
    print("updating H and creating labels")
    # Perform symmetric NMF to compute the matrix H
    H = mysymnmf.symnmf(W, H_init, len(W), len(H_init[0]))
    # Generate labels based on the index of the maximum value in each row of H
    return [np.argmax(row) for row in H]


def symnmf(X, k):
    """Executes symmetric Non-negative Matrix Factorization (NMF) on the data from a specified file.

    Args:
        filepath (str): The path to the input data file (CSV format).
        k (int): The number of components to factorize into.

    Returns:
        list: A list of labels resulting from the NMF process.
    """
    print(f"running symnmf with k = {k}")
    # Convert the input CSV file to a matrix X
    X = X.values.tolist()
    # Normalize the matrix X to create the input matrix W for NMF
    W = mysymnmf.norm(X, len(X), len(X[0]))
    # Generate the initial matrix H
    H = gen_H(W, k)
    # Generate and return labels based on the NMF result
    return symnmf_labels(W, H)
