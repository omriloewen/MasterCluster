import numpy as np
import pandas as pd
import mykmeanssp


# Get the data points from an input file
def read_file(path):
    """Read a CSV file and convert it to a list of lists.

    Args:
        path (str): The path to the CSV file from which to read the data.

    Returns:
        list: A list of lists containing the data points read from the file.
    """
    print("kmeanspp.read_file")

    return pd.read_csv(path, header=None).to_numpy().tolist()


def init_centroids(vectors, k):
    """Initialize centroids for K-Means clustering using the K-Means++ algorithm.

    The K-Means++ algorithm selects initial cluster centers that are farther apart
    to improve the convergence of the K-Means algorithm.

    Args:
        vectors (list): A list of data points (each as a list or array).
        k (int): The number of clusters (centroids) to create.

    Raises:
        ValueError: If `k` is less than 2 or greater than or equal to the number of data points.

    Returns:
        list: A list of initialized centroids.
    """
    print("kmeanspp.init_centroids")
    N = len(vectors)

    if k < 2 or k >= N:
        raise ValueError(
            "Invalid number of clusters"
        )  # Raise an exception for invalid input

    cents = [
        vectors[np.random.choice(N)].copy()
    ]  # Initialize the first centroid randomly chosen from the data points

    for _ in range(1, k):
        # Compute the distance of each point to the nearest centroid
        D = np.array(
            [
                min(np.linalg.norm(np.array(v) - np.array(c)) for c in cents)
                for v in vectors
            ]
        )
        probs = D / D.sum()  # Normalize the distances to use as probabilities
        cents.append(
            vectors[np.random.choice(N, p=probs)].copy()
        )  # Choose the next center probabilistically

    return cents


def kmeans_labels(vectors, centroids, k, iter=300, e=0.001):
    """Assign cluster labels to data points using the K-Means algorithm.

    This function uses an external implementation of K-Means clustering.

    Args:
        vectors (list): A list of data points (each as a list or array).
        centroids (list): A list of current centroids for the K-Means algorithm.
        k (int): The number of clusters (centroids).
        iter (int): The maximum number of iterations for the K-Means algorithm (default is 300).
        e (float): The threshold for convergence (default is 0.001).

    Returns:
        list: A list of cluster labels corresponding to the input data points.
    """
    print("kmeanspp.kmean_labels")
    N = len(vectors)
    d = len(
        vectors[0]
    )  # The dimensionality of the data (assumed consistent across all points)
    # Call the K-Means algorithm from the mykmeanssp module
    return mykmeanssp.fit(vectors, centroids, N, d, k, iter, e)


def kmeanspp(file_path, k):
    """Perform K-Means clustering on data read from a file.

    This function reads data from a CSV file, initializes centroids using K-Means++,
    and assigns cluster labels to the data points.

    Args:
        file_path (str): The path to the CSV file containing the data points.
        k (int): The number of clusters to create.

    Returns:
        list: A list of cluster labels corresponding to the input data points.
    """
    print("kmeanspp.kmeanspp")
    X = read_file(file_path)  # Read data points from the provided file path
    centroids = init_centroids(X, k)  # Initialize centroids for the K-Means algorithm
    return kmeans_labels(
        X, centroids, k
    )  # Assign cluster labels to the data points using the initialized centroids
