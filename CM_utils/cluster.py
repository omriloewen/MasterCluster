from . import kmeanspp
from . import symnmf
import pandas as pd
from sklearn.metrics import silhouette_score


def kmeans_cluster(X, k):
    print("kmeans_cluster")
    labels = kmeanspp.kmeanspp(X, k)
    score = silhouette_score(X.values, labels)
    return labels, score


def symnmf_cluster(X, k):
    print("symnmf_cluster")
    labels = symnmf.symnmf(X, k)
    score = silhouette_score(X.values, labels)
    return labels, score


def optimal_cluster(X, k):
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
    print("optimal_cluster")
    # Get cluster labels from KMeans++ algorithm
    kmeans_labels = kmeanspp.kmeanspp(X, k)
    # Get cluster labels from Symmetric Non-negative Matrix Factorization
    symnmf_labels = symnmf.symnmf(X, k)
    # Load the data from the specified CSV file
    X_values_np = X.values
    # Calculate silhouette scores for both clustering methods
    Kmeans_score = silhouette_score(X_values_np, kmeans_labels)
    symnmf_score = silhouette_score(X_values_np, symnmf_labels)
    # Compare silhouette scores and choose the clustering method with the higher score
    if Kmeans_score > symnmf_score:
        print("chose kmeans with score: ", Kmeans_score)
        return kmeans_labels, Kmeans_score

    else:
        print("chose symnmf with score: ", symnmf_score)
        return symnmf_labels, symnmf_score


def elbow_optimal_cluster(X, threshold, maxk):
    """Clusters the data using either k-means or SymNMF based on silhouette scores.

    Args:
        filepath (str): The path to the CSV file containing the data to be clustered.

    Returns:
        array: The cluster labels assigned to the data points based on the chosen clustering method.
    """
    print("elbow_optimal_cluster")
    # Load the data from the specified CSV file
    # Perform elbow clustering with k-means
    kmeans_res = elbow_clustering(X, kmeanspp.kmeanspp, threshold, maxk)
    # Perform elbow clustering with Symmetric Non-negative Matrix Factorization (SymNMF)
    symnmf_res = elbow_clustering(X, symnmf.symnmf, threshold, maxk)

    # Compare silhouette scores and choose the clustering method accordingly
    if kmeans_res[1] > symnmf_res[1]:
        print("Chose kmeans with score:", kmeans_res[1])
        return kmeans_res
    else:
        print("Chose SymNMF with score:", symnmf_res[1])
        return symnmf_res


def elbow_kmeans_cluster(X, threshold, maxk):
    print("elbow_kmeans_cluster")
    return elbow_clustering(X, kmeanspp.kmeanspp, threshold, maxk)


def elbow_symnmf_cluster(X, threshold, maxk):
    print("elbow_symnmf_cluster")
    return elbow_clustering(X, symnmf.symnmf, threshold, maxk)


def elbow_clustering(X, clustering_method, min_delta=0.02, maxk=50):
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
    print("elbow_clustering with method ", clustering_method)
    X_values_np = X.values
    # Generate cluster labels and silhouette score for 2 clusters
    labels2 = clustering_method(X, 2)
    score_m3 = silhouette_score(X_values_np, labels2)
    # Generate cluster labels and silhouette score for 3 clusters
    labels_m2 = clustering_method(X, 3)
    score_m2 = silhouette_score(X_values_np, labels_m2)
    # Calculate the improvement in score from 2 clusters to 3 clusters
    d_m2 = score_m2 - score_m3
    # If the improvement is not significant, return the results for 2 clusters
    if d_m2 < min_delta:
        return labels2, score_m3
    # Prepare to evaluate 4 clusters
    labels_m1 = clustering_method(X, 4)
    score_m1 = silhouette_score(X_values_np, labels_m1)

    d_m1 = score_m1 - score_m2
    # Iterate from 5 clusters up to maxk to find the optimal number of clusters
    for k in range(5, min(X.shape[0], maxk)):
        labels = clustering_method(X, k)
        score = silhouette_score(X_values_np, labels)
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
