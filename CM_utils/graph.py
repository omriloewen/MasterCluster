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
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def create_3d_cluster_plot(X, cluster_labels):
    """Creates a 3D scatter plot of clustered data.

    Args:
        X (ndarray): The input data that has been reduced to three dimensions.
                     Shape should be (n_samples, n_components).
        cluster_labels (ndarray): Labels indicating the cluster assignment
                                   for each sample in X.
    """

    # Create DataFrame from the reduced results for easier plotting
    df = pd.DataFrame(
        {
            "Component 1": X[:, 0],
            "Component 2": X[:, 1],
            "Component 3": X[:, 2],
            "Cluster": cluster_labels.astype(
                str
            ),  # Convert cluster labels to strings for plotting
            "Point Index": np.arange(X.shape[0]),  # Create a point index for hover data
        }
    )

    # Generate 3D scatter plot using Plotly
    fig = px.scatter_3d(
        df,
        x="Component 1",
        y="Component 2",
        z="Component 3",
        color="Cluster",
        title="Cluster Assignment",
        labels={"color": "Cluster"},
        width=1200,
        height=650,
        hover_data=["Point Index", "Cluster"],
    )

    # Update figure layout for dark mode for better visibility on dark backgrounds
    update_dark_mode_layout(fig)

    # Save the plot as an HTML file in the specified directory
    fig.write_html("templates/cluster_plot.html", full_html=False)


def update_dark_mode_layout(fig):
    """Updates the layout of the figure for dark mode.

    Args:
        fig (plotly.graph_objs.Figure): The Plotly figure to be updated.
    """
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"),
            yaxis=dict(backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"),
            zaxis=dict(backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"),
        ),
        paper_bgcolor="#121212",  # Set the background color of the paper
        plot_bgcolor="#121212",  # Set the background color of the plot area
        font_color="#ffffff",  # Set the font color for better readability
    )
