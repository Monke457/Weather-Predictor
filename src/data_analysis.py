import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_elbow(data, max_clusters=10):
    distortions = []  # To store the distortion (inertia) for different values of k

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20)
        kmeans.fit_predict(data)
        distortions.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method for Time Series Data')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.grid(True)
    plt.show()


def plot_clustered_data(clustered_data):
    num_clusters = len(clustered_data['cluster'].unique())

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id in range(num_clusters):
        cluster = clustered_data[clustered_data['cluster'] == cluster_id]
        ax.scatter(cluster["max_temp"], cluster["min_temp"], cluster["mean_temp"], label=f'Cluster {cluster_id}')

    ax.set_xlabel('Max Temp')
    ax.set_ylabel('Min Temp')
    ax.set_zlabel('Mean Temp')
    plt.title('Clustered Data in 3D')
    plt.legend()
    plt.show()


def analyse_data(filepath="../data/weather_processed.pkl", plot=False):
    print("running data analysis...")

    # ---------------------------------------
    # Load the data
    # ---------------------------------------
    weather = pd.read_pickle(filepath)

    # ---------------------------------------
    # K Means Clustering
    # ---------------------------------------
    weather_clustered = weather.copy()
    cluster_columns = ["max_temp", "min_temp", "mean_temp"]

    # Find the optimal number of clusters with an elbow curve
    # plot_elbow(weather[cluster_columns])

    # Assign cluster values to data
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=20)
    weather_clustered['cluster'] = kmeans.fit_predict(weather_clustered[cluster_columns])

    # Plot cluster data
    if plot:
        plot_clustered_data(weather_clustered)

    # ---------------------------------------
    # Export
    # ---------------------------------------
    weather_clustered.to_pickle("../data/weather_analysis.pkl")
    print("    weather processed -> data/weather_analysis.pkl")
    return weather_clustered

