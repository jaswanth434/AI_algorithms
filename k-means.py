import numpy as np
import matplotlib.pyplot as plt

import random

def read_data(filename):
    return np.loadtxt(filename, delimiter=',')

def generate_seed_points(data, n_clusters):
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    size_x = (x_max - x_min) / n_clusters
    size_y = (y_max - y_min) / n_clusters

    n_macroblocks = n_clusters ** 2
    avg_density = len(data) / n_macroblocks

    macroblock_densities = []

    for i in range(n_clusters):
        for j in range(n_clusters):
            x_low = x_min + i * size_x
            y_low = y_min + j * size_y
            x_high = x_low + size_x
            y_high = y_low + size_y

            points_in_macroblock = np.sum((data[:, 0] >= x_low) & (data[:, 0] < x_high) & 
                                          (data[:, 1] >= y_low) & (data[:, 1] < y_high))

            if points_in_macroblock > avg_density:
                macroblock_densities.append(((x_low + x_high) / 2, (y_low + y_high) / 2))

    seed_points = random.sample(macroblock_densities, n_clusters)
    return np.array(seed_points), min(size_x, size_y) / 2

def k_means_clustering(data, n_clusters, max_iter, epsilon):
    centroids, radius = generate_seed_points(data, n_clusters)

    clusters = [[] for _ in range(n_clusters)]
    outliers = []

    for iteration in range(max_iter):
        clusters = [[] for _ in range(n_clusters)]
        outliers = []

        for point in data:
            distances = np.linalg.norm(point - centroids, axis=1)
            nearest_cluster = np.argmin(distances)
            if distances[nearest_cluster] < radius:
                clusters[nearest_cluster].append(point)
            else:
                outliers.append(point)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                new_centroids.append(None)

        plt.figure(figsize=(10, 8))
        for cluster_index, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if cluster.size > 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {cluster_index+1}')
                plt.scatter(centroids[cluster_index][0], centroids[cluster_index][1], 
                            color='k', marker='x', s=100, lw=3)
        
        if outliers:
            outliers = np.array(outliers)
            plt.scatter(outliers[:, 0], outliers[:, 1], c='grey', label='Outliers')

        plt.title(f'Iteration {iteration + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()

        converged = True
        for idx, centroid in enumerate(centroids):
            if new_centroids[idx] is not None and np.linalg.norm(centroid - new_centroids[idx]) > epsilon:
                converged = False
                break

        if converged:
            break
        else:
            centroids = [centroid for centroid in new_centroids if centroid is not None]

    cluster_output = [(centroids[i], clusters[i]) for i in range(n_clusters) if len(clusters[i]) > 0]

    plt.figure(figsize=(10, 8))
    for cluster_index, (centroid, cluster) in enumerate(cluster_output):
        cluster = np.array(cluster)
        if cluster.size > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {cluster_index+1}')
            plt.scatter(centroid[0], centroid[1], color='k', marker='x', s=100, lw=3)

    if len(outliers) > 0: 
        outliers = np.array(outliers)
        plt.scatter(outliers[:, 0], outliers[:, 1], c='grey', label='Outliers', alpha=0.6)

    plt.title('Final Clustering Result')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

    return cluster_output, outliers


def main():
    filename = "data.txt"
    n_clusters = int(input("Enter the number of clusters: "))
    max_iter = int(input("Enter the maximum number of iterations: "))
    epsilon = float(input("Enter the maximum allowable centroid shift (epsilon): "))

    data = read_data(filename)
    centroids, outliers = k_means_clustering(data, n_clusters, max_iter, epsilon)

if __name__ == "__main__":
    main()
