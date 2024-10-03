from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np


def reduce_dimensions(X, n_components=3):
    """Reduces the dimensions of the dataset using PCA.

    Args:
        X (ndarray): The input data to reduce, where rows represent samples
                     and columns represent features.
        n_components (int): The number of principal components to return.
                            Default is 3.

    Returns:
        ndarray: The transformed dataset with reduced dimensions.
    """
    print("graph.reduce_dimensions")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def create_2d_cluster_plot(X, cluster_labels, axis_labels):
    """Creates a 2D scatter plot of clustered data.

    Args:
        X (ndarray): The input data that has been reduced to two dimensions.
                     Shape should be (n_samples, 2).
        cluster_labels (ndarray): Labels indicating the cluster assignment
                                   for each sample in X.
    """
    print("graph.create_2d_cluster_plot")
    print(axis_labels)
    print(X[:, 0].shape)
    print(X[:, 1].shape)
    print(cluster_labels.astype(str).shape)
    print(np.arange(X.shape[0]).shape)

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
    fig.write_html("templates/cluster_plot.html", full_html=False)


def create_3d_cluster_plot(X, cluster_labels, axis_labels):
    """Creates a 3D scatter plot of clustered data.

    Args:
        X (ndarray): The input data that has been reduced to three dimensions.
                     Shape should be (n_samples, n_components).
        cluster_labels (ndarray): Labels indicating the cluster assignment
                                   for each sample in X.
    """
    print("graph.create_3d_cluster_plot")
    print(axis_labels)
    print(X[:, 0].shape)
    print(X[:, 1].shape)
    print(X[:, 2].shape)
    print(cluster_labels.astype(str).shape)
    print(np.arange(X.shape[0]).shape)

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
    fig.write_html("templates/cluster_plot.html", full_html=False)


def update_dark_mode_layout(fig, is_3d=False):
    """Updates the layout of the figure for dark mode.

    Args:
        fig (plotly.graph_objs.Figure): The Plotly figure to be updated.
        is_3d (bool): If True, apply 3D-specific settings; otherwise, apply 2D settings.
    """
    print("graph.update_dark_mode_layout")

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
