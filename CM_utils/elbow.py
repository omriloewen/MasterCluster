from . import kmeanspp
from . import symnmf
import pandas as pd
from sklearn.metrics import silhouette_score


def cluster(filepath):
    """Clusters the data using either k-means or SymNMF based on silhouette scores.

    Args:
        filepath (str): The path to the CSV file containing the data to be clustered.

    Returns:
        array: The cluster labels assigned to the data points based on the chosen clustering method.
    """
    # Load the data from the specified CSV file
    X = pd.read_csv(filepath, header=None, delimiter=",")
    # Perform elbow clustering with k-means
    kmeans_res = elbow_clustering(X, kmeanspp.kmeanspp, "kmeans")
    # Perform elbow clustering with Symmetric Non-negative Matrix Factorization (SymNMF)
    symnmf_res = elbow_clustering(X, symnmf.symnmf, "SymNMF")

    # Compare silhouette scores and choose the clustering method accordingly
    if kmeans_res[1] > symnmf_res[1]:
        print("Chose kmeans with score:", kmeans_res[1])
        return kmeans_res[0]
    else:
        print("Chose SymNMF with score:", symnmf_res[1])
        return symnmf_res[0]


def elbow_clustering(X, clustering_method, maxk=50, min_delta=0.02):
    """Determines the optimal number of clusters for the provided clustering method.

    The method uses silhouette scores to evaluate the clustering quality and applies
    the elbow method to find the point where adding more clusters yields diminishing returns.

    Args:
        X (DataFrame): The input data to be clustered.
        clustering_method (callable): The clustering function to be used, e.g., kmeanspp.kmeanspp or symnmf.symnmf.
        maxk (int, optional): The maximum number of clusters to evaluate (default is 50).
        min_delta (float, optional): The minimum score difference to consider (default is 0.02).

    Returns:
        tuple: A tuple containing the labels of the clusters and the best silhouette score achieved.
    """
    # Generate cluster labels and silhouette score for 2 clusters
    labels2 = clustering_method(X, 2)
    score_m3 = silhouette_score(X, labels2)
    # Generate cluster labels and silhouette score for 3 clusters
    labels_m2 = clustering_method(X, 3)
    score_m2 = silhouette_score(X, labels_m2)
    # Calculate the improvement in score from 2 clusters to 3 clusters
    d_m2 = score_m2 - score_m3
    # If the improvement is not significant, return the results for 2 clusters
    if d_m2 < min_delta:
        return labels2, score_m3
    # Prepare to evaluate 4 clusters
    labels_m1 = clustering_method(X, 4)
    score_m1 = silhouette_score(X, labels_m1)
    d_m1 = score_m1 - score_m2
    # Iterate from 5 clusters up to maxk to find the optimal number of clusters
    for k in range(5, min(len(X), maxk)):
        labels = clustering_method(X, k)
        score = silhouette_score(X, labels)
        d = score - score_m1
        # If the score improvement conditions indicate an optimal point, return the results
        if d < d_m1 and d_m1 < d_m2:
            return labels_m2, score_m2
        else:  # Update the previous cluster results for the next iteration
            labels_m2, labels_m1 = labels_m1, labels
            score_m2, score_m1 = score_m1, score
            d_m2, d_m1 = d_m1, d
    # If no optimal point found, return the results for the last clustering attempt
    return labels, score
