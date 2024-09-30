from . import kmeanspp
from . import symnmf
import pandas as pd
from sklearn.metrics import silhouette_score


def cluster(filepath, k):
    """Clusters data from a CSV file using KMeans++ and Symmetric Non-negative Matrix Factorization (SymNMF).

    This function reads the dataset from the specified file, applies two different clustering
    algorithms (KMeans++ and SymNMF), and evaluates their performances using the silhouette score.
    The clustering method with the higher silhouette score is selected.

    Args:
        filepath (str): The path to the CSV file containing the dataset to be clustered.
        k (int): The number of clusters to form.

    Returns:
        array-like: The labels of the clusters based on the selected clustering algorithm.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If k is less than 1.
    """
    print("cluster.cluster")
    # Get cluster labels from KMeans++ algorithm
    kmeans_labels = kmeanspp.kmeanspp(filepath, k)
    # Get cluster labels from Symmetric Non-negative Matrix Factorization
    symnmf_labels = symnmf.symnmf(filepath, k)
    # Load the data from the specified CSV file
    X = pd.read_csv(filepath, header=None).to_numpy()
    # Calculate silhouette scores for both clustering methods
    Kmeans_score = silhouette_score(X, kmeans_labels)
    symnmf_score = silhouette_score(X, symnmf_labels)
    # Compare silhouette scores and choose the clustering method with the higher score
    if Kmeans_score > symnmf_score:
        print("chose kmeans with score: ", Kmeans_score)
        return kmeans_labels

    else:
        print("chose symnmf with score: ", symnmf_score)
        return symnmf_labels
