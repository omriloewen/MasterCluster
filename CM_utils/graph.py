from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np


def reduce_dimensions(X, n_components=3):
    """Reduces the dimensions of the dataset using Principal Component Analysis (PCA).

    This function transforms the original dataset into a lower-dimensional space
    defined by the principal components, which capture the most variance in the data.

    Args:
        X (ndarray): The input data to reduce, where rows represent samples
                     and columns represent features.
        n_components (int): The number of principal components to return.
                            Default is 3.

    Returns:
        ndarray: The transformed dataset with reduced dimensions (shape (n_samples, n_components)).
    """
    print("reducing dimensions")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def create_2d_cluster_plot(X, cluster_labels, axis_labels, graph_path):
    """Creates a 2D scatter plot of clustered data.

    The function visualizes the clusters in a 2D plane, using different colors
    to represent different clusters. It can also display the index of each point
    in the hover information.

    Args:
        X (ndarray): The input data that has been reduced to two dimensions,
                     expected shape is (n_samples, 2).
        cluster_labels (ndarray): Labels indicating the cluster assignment
                                   for each sample in X.
        axis_labels (list of str): A list containing labels for the x-axis and y-axis.
        graph_path (str): Path where the generated plot will be saved as an HTML file.

    Returns:
        None
    """
    print("creating 2d cluster plot")

    # Create DataFrame from the reduced results for easier plotting
    df = pd.DataFrame(
        {
            axis_labels[0]: X[:, 0],
            axis_labels[1]: X[:, 1],
            "Cluster": cluster_labels.astype(
                str
            ),  # Convert cluster labels to strings for plotting
            "Point Index": np.arange(X.shape[0]),  # Create a point index for hover data
        }
    )

    # Generate 2D scatter plot using Plotly
    fig = px.scatter(
        df,
        x=axis_labels[0],
        y=axis_labels[1],
        color="Cluster",
        title="",
        labels={"color": "Cluster"},
        width=1000,
        height=600,
        hover_data=["Point Index", "Cluster"],
    )

    # Update figure layout for dark mode for better visibility on dark backgrounds
    update_dark_mode_layout(fig)

    # Save the plot as an HTML file in the specified directory
    fig.write_html(graph_path, full_html=False)


def create_3d_cluster_plot(X, cluster_labels, axis_labels, graph_path):
    """Creates a 3D scatter plot of clustered data.

    This function visualizes the clusters in a 3D space, using different colors
    to represent different clusters. It can also display the index of each point
    in the hover information.

    Args:
        X (ndarray): The input data that has been reduced to three dimensions,
                     expected shape is (n_samples, 3).
        cluster_labels (ndarray): Labels indicating the cluster assignment
                                   for each sample in X.
        axis_labels (list of str): A list containing labels for the x-axis, y-axis, and z-axis.
        graph_path (str): Path where the generated plot will be saved as an HTML file.

    Returns:
        None
    """
    print("creating 3d cluster plot")

    # Create DataFrame from the reduced results for easier plotting
    df = pd.DataFrame(
        {
            axis_labels[0]: X[:, 0],
            axis_labels[1]: X[:, 1],
            axis_labels[2]: X[:, 2],
            "Cluster": cluster_labels.astype(
                str
            ),  # Convert cluster labels to strings for plotting
            "Point Index": np.arange(X.shape[0]),  # Create a point index for hover data
        }
    )

    # Generate 3D scatter plot using Plotly
    fig = px.scatter_3d(
        df,
        x=axis_labels[0],
        y=axis_labels[1],
        z=axis_labels[2],
        color="Cluster",
        title="",
        labels={"color": "Cluster"},
        width=1000,
        height=600,
        hover_data=["Point Index", "Cluster"],
    )

    # Update figure layout for dark mode for better visibility on dark backgrounds
    update_dark_mode_layout(fig, True)

    # Save the plot as an HTML file in the specified directory
    fig.write_html(graph_path, full_html=False)


def update_dark_mode_layout(fig, is_3d=False):
    """Updates the layout of the figure for dark mode.

    This function modifies the layout properties of the given Plotly figure
    to improve visibility against dark backgrounds, specifically designed for
    both 2D and 3D plots.

    Args:
        fig (plotly.graph_objs.Figure): The Plotly figure to be updated.
        is_3d (bool): If True, apply 3D-specific settings; otherwise, apply 2D settings.

    Returns:
        None
    """
    print("updating plot dark mode layout")

    if is_3d:
        # Apply dark mode styling for 3D plots
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"
                ),
                yaxis=dict(
                    backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"
                ),
                zaxis=dict(
                    backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"
                ),
            ),
            paper_bgcolor="#121212",  # Set the background color of the paper
            plot_bgcolor="#121212",  # Set the background color of the plot area
            font_color="#ffffff",  # Set the font color for better readability
        )
    else:
        # Apply dark mode styling for 2D plots
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridcolor="#444444",
                zerolinecolor="#444444",
                color="#ffffff",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#444444",
                zerolinecolor="#444444",
                color="#ffffff",
            ),
            paper_bgcolor="#121212",  # Set the background color of the paper
            plot_bgcolor="#121212",  # Set the background color of the plot area
            font_color="#ffffff",  # Set the font color for better readability
        )
