import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from data_transformation import PrincipalComponentAnalysis


def analyse_data(filepath="../data/weather_processed.pkl", plot=False):
    print("running data analysis...")
    weather = pd.read_pickle(filepath)

    # ---------------------------------------
    # K Means Clustering
    # ---------------------------------------
    weather_clustered = weather.copy()
    cluster_columns = ["max_temp", "min_temp", "mean_temp"]

    # Find the optimal number of clusters with an elbow curve
    if plot:
        plot_elbow_cluster(weather[cluster_columns])

    # Assign cluster values to data
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=20)
    weather_clustered['cluster'] = kmeans.fit_predict(
        weather_clustered[cluster_columns])

    # Plot cluster data
    if plot:
        plot_clustered_data(weather_clustered, cluster_columns)

    # ---------------------------------------
    # Principal Component Analysis
    # ---------------------------------------
    weather_pca = weather_clustered.copy()
    pca = PrincipalComponentAnalysis()
    predictor_columns = ["max_temp", "min_temp", "mean_temp", "sunshine",
                         "cloud_cover", "precipitation", "pressure"]

    # Find the optimal number of pca components
    pc_values = pca.determine_pc_explained_variance(
        weather_pca, predictor_columns)
    if plot:
        plot_elbow(pc_values, len(predictor_columns),
                   'Number of components', 'Variance')

    weather_pca = pca.apply_pca(weather_pca, predictor_columns, 4)
    if plot:
        plot_pca(weather_pca)

    weather_pca.to_pickle(filepath)
    print(f"    weather analysis complete -> {filepath}")
    return weather_pca


def plot_elbow(data, column_size, x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, column_size + 1), data, marker='o')
    plt.title('Elbow Method')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()


def plot_elbow_cluster(data, max_clusters=10):
    distortions = []  # To store the distortion (inertia) for different clusters

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20)
        kmeans.fit_predict(data)
        distortions.append(kmeans.inertia_)

    plot_elbow(distortions, max_clusters, 'Number of clusters',
               'Distortion (Inertia)')


def plot_clustered_data(clustered_data, columns):
    num_clusters = len(clustered_data['cluster'].unique())

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id in range(num_clusters):
        cluster = clustered_data[clustered_data['cluster'] == cluster_id]
        ax.scatter(cluster[columns[0]], cluster[columns[1]],
                   cluster[columns[2]], label=f'Cluster {cluster_id}')

    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    plt.title('Clustered Data in 3D')
    plt.legend()
    plt.show()


def plot_pca(data):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x=data.index, y="pca_1", label="PCA 1")
    sns.lineplot(data=data, x=data.index, y="pca_2", label="PCA 2")
    sns.lineplot(data=data, x=data.index, y="pca_3", label="PCA 3")
    sns.lineplot(data=data, x=data.index, y="pca_4", label="PCA 4")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("PCA Values 1979 - 2020")
    plt.legend()
    plt.show()
