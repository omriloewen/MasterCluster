from sklearn.decomposition import PCA
import plotly.express as px


def reduce_dimensions(X, n_components=3):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced


import plotly.express as px
import pandas as pd
import numpy as np


def create_3d_cluster_plot(X, cluster_labels):
    # יצירת אינדקסים לנקודות
    indices = np.arange(X.shape[0])

    # יצירת DataFrame עם הנתונים
    df = pd.DataFrame(
        {
            "Component 1": X[:, 0],
            "Component 2": X[:, 1],
            "Component 3": X[:, 2],
            "cluster": cluster_labels.astype(str),
            "point": indices,
        }
    )

    # יצירת הגרף עם Plotly
    fig = px.scatter_3d(
        df,
        x="Component 1",
        y="Component 2",
        z="Component 3",
        color="cluster",
        title="Cluster Assignment",
        labels={
            "Component 1": "Component 1",
            "Component 2": "Component 2",
            "Component 3": "Component 3",
            "color": "cluster",  # שינוי תווית הצבע
        },
        width=1200,
        height=650,
        hover_data=["point", "cluster"],  # הוספת אינדקס ומספר קבוצה לריחוף
    )

    # עדכון עיצוב הגרף למצב לילה
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"),
            yaxis=dict(backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"),
            zaxis=dict(backgroundcolor="#1e1e1e", color="#ffffff", gridcolor="#444444"),
        ),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font_color="#ffffff",
    )

    # שמירת הגרף כקובץ HTML
    fig.write_html("templates/cluster_plot.html", full_html=False)
