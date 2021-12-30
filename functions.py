import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from typing import List

matplotlib.use('TkAgg')


def k_means(
        number_of_cluster: int,
        y: str,
        x: str,
        file_dir: str,
        visualization_colors: List[str]
) -> pd.DataFrame:

    csv_data = pd.read_csv(file_dir)

    data = csv_data[[x, y]]

    # Select random observation as centroids
    centroids = (data.sample(n=number_of_cluster))
    plt.scatter(data[x], data[y], c='black')
    plt.scatter(centroids[x], centroids[y], c='red')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

    new_centroids = pd.DataFrame()
    clustering_data = pd.DataFrame()

    while not centroids.equals(new_centroids):

        # check if `new_centroids` is not empty to handle first iteration in loop
        if not new_centroids.empty:
            centroids = new_centroids

        # a variable to save distance and other calculation data
        clustering_data = data

        i = 1

        # calculate distance of each item to others
        for _, row_c in centroids.iterrows():
            destinations = []
            for _, row_d in data.iterrows():
                d1 = (row_c[x] - row_d[x]) ** 2
                d2 = (row_c[y] - row_d[y]) ** 2
                d = np.sqrt(d1 + d2)
                destinations.append(d)
            clustering_data[i] = destinations
            i = i + 1

        cluster = []
        # choose closest centroid
        for index, row in clustering_data.iterrows():
            min_dist = row[1]
            pos = 1
            for i in range(number_of_cluster):
                if row[i + 1] < min_dist:
                    min_dist = row[i + 1]
                    pos = i + 1
            cluster.append(pos)
        clustering_data["Cluster"] = cluster

        # calculate new centroids for each cluster
        new_centroids = clustering_data.groupby(["Cluster"]).mean()[[y, x]]

        # visualize clustering result
        for k in range(number_of_cluster):
            visualize_data = clustering_data[clustering_data["Cluster"] == k + 1]
            plt.scatter(visualize_data[x], visualize_data[y], c=visualization_colors[k])
        plt.scatter(centroids[x], centroids[y], c='red')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    # final picture of clustering result
    plt.show()
    csv_data["Cluster"] = clustering_data["Cluster"]
    return csv_data


def calculate_accuracy(df: pd.DataFrame, number_of_clusters):
    items_with_correct_type = 0
    for i in range(number_of_clusters):
        type_value = df[df.Cluster == i + 1].mode()["Type"][0]
        items_with_correct_type += len(df[(df.Cluster == i + 1) & (df.Type == type_value)].index)

    all_items_count = len(df.index)

    return items_with_correct_type / all_items_count * 100
