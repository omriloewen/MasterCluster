from . import kmeanspp
from . import symnmf
import pandas as pd
from sklearn.metrics import silhouette_score


def kmeans_cluster(X, k):
    """Apply KMeans++ clustering to the dataset.

    This function applies the KMeans++ algorithm to the input data and calculates
    the silhouette score to evaluate the clustering performance.

    Args:
        X (pd.DataFrame): The input data to cluster.
        k (int): The number of clusters.

    Returns:
        tuple: A tuple containing the cluster labels and the silhouette score.
    """
    print(f"running kmeans clustering with k = {k}")
    labels = kmeanspp.kmeanspp(X, k)
    score = silhouette_score(X.values, labels)
    return labels, score


def symnmf_cluster(X, k):
    """Apply Symmetric Non-negative Matrix Factorization (SymNMF) clustering.

    This function applies the SymNMF algorithm to the input data and calculates
    the silhouette score to evaluate the clustering performance.

    Args:
        X (pd.DataFrame): The input data to cluster.
        k (int): The number of clusters.

    Returns:
        tuple: A tuple containing the cluster labels and the silhouette score.
    """
    print(f"running symnmf clustering with k = {k}")
    labels = symnmf.symnmf(X, k)
    score = silhouette_score(X.values, labels)
    return labels, score


def optimal_cluster(X, k):
    """Select the optimal clustering method between KMeans++ and SymNMF.

    This function compares the clustering results from KMeans++ and SymNMF
    algorithms based on their silhouette scores and returns the labels of
    the clusters from the better-performing method.

    Args:
        X (pd.DataFrame): Data to be clustered.
        k (int): The number of clusters to form.

    Returns:
        tuple: A tuple containing the labels of the selected clusters and the corresponding silhouette score.

    Raises:
        ValueError: If k is less than 1.
    """
    print(f"running optimal clustering with k = {k}")
    kmeans_labels = kmeanspp.kmeanspp(X, k)  # Apply KMeans++
    symnmf_labels = symnmf.symnmf(X, k)  # Apply SymNMF
    X_values_np = X.values  # Convert DataFrame to numpy array
    # Calculate silhouette scores for both algorithms
    Kmeans_score = silhouette_score(X_values_np, kmeans_labels)
    symnmf_score = silhouette_score(X_values_np, symnmf_labels)
    # Select the clustering method with the higher silhouette score
    if Kmeans_score > symnmf_score:
        print("chose kmeans with score: ", Kmeans_score)
        return kmeans_labels, Kmeans_score

    else:
        print("chose symnmf with score: ", symnmf_score)
        return symnmf_labels, symnmf_score


def elbow_optimal_cluster(X, threshold, maxk):
    """Perform elbow method to find the optimal number of clusters.

    This function applies both KMeans and SymNMF clustering methods,
    evaluates their performances based on silhouette scores, and
    selects the method with the best score.

    Args:
        X (pd.DataFrame): Data to be clustered.
        threshold (float): The threshold for silhouette score improvement.
        maxk (int): The maximum number of clusters to consider.

    Returns:
        tuple: A tuple containing the cluster labels of the selected method and the corresponding silhouette score.
    """
    print("running optimal clustering with elbow method")
    kmeans_res = elbow_clustering(
        X, kmeanspp.kmeanspp, threshold, maxk
    )  # Elbow for KMeans
    symnmf_res = elbow_clustering(X, symnmf.symnmf, threshold, maxk)  # Elbow for SymNMF

    # Compare silhouette scores and choose the clustering method accordingly
    if kmeans_res[1] > symnmf_res[1]:
        print("Chose kmeans with score:", kmeans_res[1])
        return kmeans_res
    else:
        print("Chose SymNMF with score:", symnmf_res[1])
        return symnmf_res


def elbow_kmeans_cluster(X, threshold, maxk):
    """Cluster data using the elbow method with KMeans.

    Args:
        X (pd.DataFrame): Data to be clustered.
        threshold (float): The threshold for silhouette score improvement.
        maxk (int): The maximum number of clusters to consider.

    Returns:
        tuple: The results of the elbow clustering using KMeans.
    """
    print("running kmeans clustering with elbow method")
    return elbow_clustering(X, kmeanspp.kmeanspp, threshold, maxk)


def elbow_symnmf_cluster(X, threshold, maxk):
    """Cluster data using the elbow method with SymNMF.

    Args:
        X (pd.DataFrame): Data to be clustered.
        threshold (float): The threshold for silhouette score improvement.
        maxk (int): The maximum number of clusters to consider.

    Returns:
        tuple: The results of the elbow clustering using SymNMF.
    """
    print("running symnmf clustering with elbow method")
    return elbow_clustering(X, symnmf.symnmf, threshold, maxk)


def elbow_clustering(X, clustering_method, e=0.02, maxk=50):
    """Apply the elbow method to determine the optimal number of clusters.

    This function evaluates the clustering method's performance based on
    silhouette scores and determines when to stop based on the provided threshold.

    Args:
        X (pd.DataFrame): The input data to cluster.
        clustering_method (function): The clustering algorithm to apply.
        e (float): The threshold for silhouette score improvement.
        maxk (int): The maximum number of clusters to consider.

    Returns:
        tuple: The cluster labels and the silhouette score of the selected number of clusters.
    """
    print(f"running elbow clustering with method {clustering_method}")
    X_values_np = X.values  # Convert DataFrame to numpy array
    labels_q = []
    score_q = []
    # Evaluate clustering for k from 2 to 5
    for k in range(2, 6):
        labels_q.append(clustering_method(X, k))
        score_q.append(silhouette_score(X_values_np, labels_q[k - 2]))
    # Evaluate clustering for k from 5 to maxk
    for k in range(5, maxk):
        # Check for changes in silhouette scores
        if score_q[1] - score_q[0] < e:
            if score_q[2] - score_q[0] < e or score_q[2] - score_q[1] < e:
                if score_q[3] - score_q[0] < e or score_q[3] - score_q[2] < e:
                    return labels_q[0], score_q[0]
        # Continue evaluating for the next k
        labels_q.append(clustering_method(X, k))
        score_q.append(silhouette_score(X_values_np, labels_q[4]))
        labels_q.pop(0)
        score_q.pop(0)

    # if no elbow point found, Compare the last scores and return the results based on threshold
    if score_q[1] - score_q[0] < e:
        return labels_q[0], score_q[0]
    if score_q[2] - score_q[1] < e:
        return labels_q[1], score_q[1]
    if score_q[3] - score_q[2] < e:
        return labels_q[2], score_q[2]
    return labels_q[3], score_q[3]
