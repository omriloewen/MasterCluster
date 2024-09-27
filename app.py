from flask import Flask, request, redirect, url_for, render_template
from CM_utils import cluster, elbow, graph
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# define the path for the file upload
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# check if the file is csv
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "csv"


def process_input(filepath, param):
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
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    auto_select = request.form.get("auto_select")
    param = request.form.get("param")
    if not (param and param.isdigit() and int(param) >= 2):
        param = 0
    if auto_select:
        param = 0
    param = int(param)
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)  # saving the file in the directory
        print(f"File successfully uploaded with parameter {param}")
    else:
        return "Invalid file type. Only CSV files are allowed"

    try:
        labels, k = process_input(filepath, param)
        labels = np.array(labels)
        X = pd.read_csv(filepath, header=None, delimiter=",")
        X = X.to_numpy()
        X_reduced = graph.reduce_dimensions(X, n_components=3)
        graph.create_3d_cluster_plot(X_reduced, labels)
        return render_template("result.html")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} has been deleted.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
