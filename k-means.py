import numpy as np
import matplotlib.pyplot as plt

import random

def read_data(filename):
    """
    Reads data points from a file.
    """
    return np.loadtxt(filename, delimiter=',')

def generate_seed_points(data, n_clusters):
    # Calculate the bounding box of the data
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])

    # Determine the size of each macroblock
    size_x = (x_max - x_min) / n_clusters
    size_y = (y_max - y_min) / n_clusters

    # Determine the number of macroblocks and average density
    n_macroblocks = n_clusters ** 2
    avg_density = len(data) / n_macroblocks

    # Initialize the macroblock density list
    macroblock_densities = []

    # Calculate densities and add macroblock centers to the list if the density is high
    for i in range(n_clusters):
        for j in range(n_clusters):
            x_low = x_min + i * size_x
            y_low = y_min + j * size_y
            x_high = x_low + size_x
            y_high = y_low + size_y

            # Calculate the number of points in the current macroblock
            points_in_macroblock = np.sum((data[:, 0] >= x_low) & (data[:, 0] < x_high) & 
                                          (data[:, 1] >= y_low) & (data[:, 1] < y_high))

            # If the macroblock's density is higher than average, add its center to the list
            if points_in_macroblock > avg_density:
                macroblock_densities.append(((x_low + x_high) / 2, (y_low + y_high) / 2))

    # Randomly select n_clusters centers from the high-density macroblocks
    seed_points = random.sample(macroblock_densities, n_clusters)
    return np.array(seed_points), min(size_x, size_y) / 2


def assign_points_to_clusters(data, centroids):
    """
    Assigns each data point to the nearest cluster.
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, assignments, n_clusters):
    """
    Updates the centroids based on the current assignment of points.
    """
    return np.array([data[assignments == k].mean(axis=0) for k in range(n_clusters)])

def k_means_clustering(data, n_clusters, max_iter, epsilon):
    centroids, radius = generate_seed_points(data, n_clusters)
    # Initialize the clusters and the set of outliers
    clusters = [[] for _ in range(n_clusters)]
    outliers = []

    for iteration in range(max_iter):
        # Reset clusters and outliers
        clusters = [[] for _ in range(n_clusters)]
        outliers = []

        # Assign points to the nearest centroid or mark as outlier
        for point in data:
            distances = np.linalg.norm(point - centroids, axis=1)
            nearest_cluster = np.argmin(distances)
            if distances[nearest_cluster] < radius:
                clusters[nearest_cluster].append(point)
            else:
                outliers.append(point)

        # Recalculate centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                # Handle the case where a cluster has no points
                new_centroids.append(None)

        plt.figure(figsize=(10, 8))
        for cluster_index, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if cluster.size > 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {cluster_index+1}')
                plt.scatter(centroids[cluster_index][0], centroids[cluster_index][1], 
                            color='k', marker='x', s=100, lw=3)
        
        # Plot outliers
        if outliers:
            outliers = np.array(outliers)
            plt.scatter(outliers[:, 0], outliers[:, 1], c='grey', label='Outliers')

        plt.title(f'Iteration {iteration + 1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()
        # Check for convergence
        converged = True
        for idx, centroid in enumerate(centroids):
            if new_centroids[idx] is not None and np.linalg.norm(centroid - new_centroids[idx]) > epsilon:
                converged = False
                break

        if converged:
            break
        else:
            centroids = [centroid for centroid in new_centroids if centroid is not None]

    # Convert clusters to the required output format
    cluster_output = [(centroids[i], clusters[i]) for i in range(n_clusters) if len(clusters[i]) > 0]

     # Plot the final clusters and centroids
    plt.figure(figsize=(10, 8))
    for cluster_index, (centroid, cluster) in enumerate(cluster_output):
        cluster = np.array(cluster)
        if cluster.size > 0:
            plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {cluster_index+1}')
            plt.scatter(centroid[0], centroid[1], color='k', marker='x', s=100, lw=3)

    # Plot outliers
    if len(outliers) > 0:  # Check if the list of outliers is not empty
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
    # outlier_dist = float(input("Enter the distance threshold to identify outliers: "))

    data = read_data(filename)
    centroids, outliers = k_means_clustering(data, n_clusters, max_iter, epsilon)

    # print("Final centroids:\n", centroids)
    # print("Cluster assignments:", assignments)


if __name__ == "__main__":
    main()
