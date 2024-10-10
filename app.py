from flask import Flask, request, redirect, url_for, render_template
from flask_caching import Cache
from CM_utils import cluster, graph
import os
import pandas as pd
import numpy as np


app = Flask(__name__)

app.config["CACHE_TYPE"] = "simple"
cache = Cache(app)

# define the path for the file upload
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create the upload directory if it does not exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
    print(f"Directory {app.config['UPLOAD_FOLDER']} created.")
else:
    print(f"Directory {app.config['UPLOAD_FOLDER']} already exists.")


# check if the file is csv
def allowed_file(filename):
    """Check if the uploaded file is allowed.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file is a CSV; False otherwise.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


def main_cluster(
    X, k, cluster_method, clustering_evaluation_method, elbow, threshold, maxk
):
    """Process the input file for clustering.

    Args:
        X (pandas data frame): the data from the input CSV file.
        k (int): The number of clusters; 0 for automatic selection.

    Returns:
        tuple: A tuple containing:
            - labels (list): Cluster labels for each data point.
            - k (int): Number of clusters.
    """
    print("main_cluster")

    if k == 0 or elbow:
        if cluster_method == "optimal":
            labels_and_score = cluster.elbow_optimal_cluster(X, threshold, maxk)
        if cluster_method == "kmeans++":
            labels_and_score = cluster.elbow_kmeans_cluster(X, threshold, maxk)
        if cluster_method == "symnmf":
            labels_and_score = cluster.elbow_symnmf_cluster(X, threshold, maxk)

    else:
        if cluster_method == "optimal":
            labels_and_score = cluster.optimal_cluster(X, k)
        if cluster_method == "kmeans++":
            labels_and_score = cluster.kmeans_cluster(X, k)
        if cluster_method == "symnmf":
            labels_and_score = cluster.symnmf_cluster(X, k)

    k = max(k, max(labels_and_score[0]) + 1)

    return labels_and_score[0], labels_and_score[1], k


def create_result_page(
    X, k, labels, csv_header, graph_dimension, use_pca, graph_attributes
):
    print("create_result_page")
    threeD = False
    labels = np.array(labels)  # Convert labels to numpy array
    headings = X.columns.tolist()

    if use_pca:
        threeD = graph_dimension == 3
        X_reduced = graph.reduce_dimensions(X.values, n_components=graph_dimension)
        graph_labels = [f"principal component {i}" for i in range(graph_dimension)]
    # Dimensionality reduction
    else:
        threeD = len(graph_attributes) == 3
        if csv_header == 0:
            graph_labels = [headings[i] for i in graph_attributes]
            X_reduced = X[graph_labels].values
        if csv_header == None:
            graph_labels = [f"component {i}" for i in graph_attributes]
            X_reduced = X.iloc[:, graph_attributes].values

    if threeD:
        graph.create_3d_cluster_plot(X_reduced, labels, graph_labels)  # Create 3D plot
    else:
        graph.create_2d_cluster_plot(X_reduced, labels, graph_labels)


@app.route("/")
def index():
    """Render the home page.

    Returns:
        str: Rendered HTML for the index page.
    """
    print("initializing home page")
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and perform clustering.

    Returns:
        str: Rendered HTML for the result page or error message.
    """
    print("Processing upload")
    csv_header = None
    graph_dimension = 3
    use_pca = True
    graph_attributes = [1, 2, 3]
    cluster_method = "optimal"
    clustering_evaluation_method = "silhouette"
    elbow = True

    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Validate file type and save it
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)  # Save the file in the upload directory
        print("File successfully uploaded with parameter")
    else:
        return "Invalid file type. Only CSV files are allowed"

    # Basic Inputs
    elbow = "auto_select" in request.form
    k = request.form.get("param") if not elbow else 0
    if not k:
        k = 0
        elbow = True
    k = int(k)

    # Advanced Inputs (Always Available)
    csv_header = (
        0 if "csv_header" in request.form else None
    )  # Checkbox is checked or unchecked
    cluster_method = request.form["cluster_method"]
    clustering_evaluation_method = request.form["eval_method"]
    graph_dimension = int(request.form["graph_dimension"])
    graph_attributes = [i for i in range(graph_dimension)]
    use_pca = "use_pca" in request.form

    # Conditional Threshold (only if auto clusters are selected)
    threshold = float(request.form["threshold"]) if elbow else 0.002
    maxk = int(request.form["max_clusters"]) if elbow else 50

    try:
        X = pd.read_csv(filepath, header=csv_header)
        # Process the input file
        labels, score, k = main_cluster(
            X, k, cluster_method, clustering_evaluation_method, elbow, threshold, maxk
        )
        create_result_page(
            X,
            k,
            labels,
            csv_header,
            graph_dimension,
            use_pca,
            graph_attributes,
        )
        X_json = X.to_json()
        if csv_header == 0:
            headers = X.columns.tolist()

        if csv_header == None:
            headers = [f"component {i}" for i in range(X.shape[1])]
        return render_template(
            "result.html",
            X_json=X_json,
            orig_k=str(k),
            labels=labels,
            orig_cluster_method=cluster_method,
            orig_clustering_evaluation_method=clustering_evaluation_method,
            orig_elbow=str(elbow),
            orig_threshold=str(threshold),
            orig_maxk=str(maxk),
            csv_header=str(csv_header),
            orig_graph_dimension=str(graph_dimension),
            orig_use_pca=str(use_pca),
            orig_graph_attributes=graph_attributes,
            D=X.shape[1],
            headers=headers,
            score=str(score),
        )  # Render the result page
    except Exception as e:
        print(e)  # Print error message
    finally:
        # Clean up by deleting the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} has been deleted.")


@app.route("/reprocess", methods=["POST"])
def reprocess():
    cache.clear()
    ReVisualize = False
    ReCluster = False
    orig_cluster_method = request.form.get("orig_cluster_method")
    cluster_method = request.form.get("cluster_method")
    orig_elbow = request.form.get("orig_elbow") == "True"
    elbow = request.form.get("elbow") == "on"
    orig_clustering_evaluation_method = request.form.get(
        "orig_clustering_evaluation_method"
    )
    clustering_evaluation_method = request.form.get("clustering_evaluation_method")
    orig_threshold = request.form.get("orig_threshold")
    threshold = request.form.get("threshold")
    orig_maxk = request.form.get("orig_maxk")
    maxk = request.form.get("maxk")
    orig_k = request.form.get("orig_k")
    k = request.form.get("k")
    orig_graph_dimension = request.form.get("orig_graph_dimension")
    graph_dimension = request.form.get("graph_dimension")
    orig_use_pca = request.form.get("orig_use_pca") == "True"
    use_pca = request.form.get("use_pca") == "on"
    orig_graph_attributes = request.form.get("orig_graph_attributes").split(",")
    graph_attributes = request.form.getlist("graph_attributes")
    df_json = request.form.get("df_json")
    csv_header = request.form.get("csv_header")
    headers = request.form.get("headers").split(",")
    labels = request.form.get("labels").split(",")
    score = request.form.get("score")

    if use_pca != orig_use_pca:
        ReVisualize = True
    else:
        if use_pca:
            ReVisualize = orig_graph_dimension != graph_dimension
        else:
            ReVisualize = orig_graph_attributes != graph_attributes

    if orig_elbow != elbow:
        ReCluster = True
    else:
        if elbow:
            ReCluster = threshold != orig_threshold or maxk != orig_maxk
        else:
            ReCluster = k != orig_k

    if (
        orig_cluster_method != cluster_method
        or orig_clustering_evaluation_method != clustering_evaluation_method
    ):
        ReCluster = True

    X = pd.read_json(df_json)
    graph_attributes = list(map(int, graph_attributes))

    if ReCluster:
        k = int(k)
        threshold = float(threshold)
        maxk = int(maxk)
        labels, score, k = main_cluster(
            X, k, cluster_method, clustering_evaluation_method, elbow, threshold, maxk
        )
        csv_header = 0 if csv_header == "0" else None
        graph_dimension = int(graph_dimension)
        create_result_page(
            X,
            k,
            labels,
            csv_header,
            graph_dimension,
            use_pca,
            graph_attributes,
        )

    if not ReCluster and ReVisualize:
        k = int(k)
        csv_header = 0 if csv_header == "0" else None
        graph_dimension = int(graph_dimension)
        create_result_page(
            X,
            k,
            labels,
            csv_header,
            graph_dimension,
            use_pca,
            graph_attributes,
        )

    return render_template(
        "result.html",
        X_json=df_json,
        orig_k=str(k),
        labels=labels,
        orig_cluster_method=cluster_method,
        orig_clustering_evaluation_method=clustering_evaluation_method,
        orig_elbow=str(elbow),
        orig_threshold=str(threshold),
        orig_maxk=str(maxk),
        csv_header=str(csv_header),
        orig_graph_dimension=str(graph_dimension),
        orig_use_pca=str(use_pca),
        orig_graph_attributes=graph_attributes,
        D=X.shape[1],
        headers=headers,
        score=str(score),
    )  # Render the result page


if __name__ == "__main__":
    # Start the Flask application
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)
