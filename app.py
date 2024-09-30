from flask import Flask, request, redirect, url_for, render_template
from CM_utils import cluster, elbow, graph
import os
import pandas as pd
import numpy as np


app = Flask(__name__)

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


def process_input(filepath, param):
    """Process the input file for clustering.

    Args:
        filepath (str): The path to the input CSV file.
        param (int): The number of clusters; 0 for automatic selection.

    Returns:
        tuple: A tuple containing:
            - labels (list): Cluster labels for each data point.
            - k (int): Number of clusters.
    """
    if param == 0:
        print("using elbow method")
        labels = elbow.cluster(filepath)
    else:
        print("clustering into ", param, " clusters")
        labels = cluster.cluster(filepath, param)

    k = max(param, max(labels) + 1)

    return labels, k


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
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    auto_select = request.form.get("auto_select")  # Check for auto selection
    param = request.form.get("param")  # Get number of clusters from the form

    # Validate and set parameter for clustering
    if not (param and param.isdigit() and int(param) >= 2):
        param = 0
    if auto_select:
        param = 0
    param = int(param)

    # Validate file type and save it
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)  # Save the file in the upload directory
        print(f"File successfully uploaded with parameter {param}")
    else:
        return "Invalid file type. Only CSV files are allowed"

    try:
        # Process the input file
        labels, k = process_input(filepath, param)
        labels = np.array(labels)  # Convert labels to numpy array
        X = pd.read_csv(filepath, header=None, delimiter=",")  # Read the CSV file
        X = X.to_numpy()  # Convert DataFrame to numpy array
        X_reduced = graph.reduce_dimensions(
            X, n_components=3
        )  # Dimensionality reduction
        graph.create_3d_cluster_plot(X_reduced, labels)  # Create 3D plot
        return render_template("result.html")  # Render the result page
    except Exception as e:
        print(e)  # Print error message
    finally:
        # Clean up by deleting the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} has been deleted.")


if __name__ == "__main__":
    # Start the Flask application
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)
