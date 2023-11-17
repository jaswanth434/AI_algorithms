import numpy as np
import random

def generate_random_data(num_points, num_clusters):
    """
    Generates random data points within specified clusters and adds 5% outliers.
    """
    data = []
    num_outliers = 1 # int(0.05 * num_points)  # 5% of total points as outliers

    # Generate cluster data
    for iteration in range(num_clusters):
        # Create a random centroid for each cluster
        center_x, center_y = random.uniform(-100, 100), random.uniform(-100, 100)
        
        # Generate points around the centroid
        for _ in range(num_points // num_clusters):
            x = np.random.normal(center_x, 10)  # Standard deviation for clusters
            y = np.random.normal(center_y, 10)
            data.append([x, y])

    # Generate outliers
    for _ in range(num_outliers):
        outlier_x = random.uniform(-100, -120)  # Larger range for outliers
        outlier_y = random.uniform(-120, 120)
        data.append([outlier_x, outlier_y])

    # Randomize the order of points
    random.shuffle(data)
    return data


def write_data_to_file(data, filename="data.txt"):
    """
    Writes the data points to a file.
    """
    with open(filename, 'w') as file:
        for point in data:
            file.write(f"{point[0]},{point[1]}\n")

def main():
    num_clusters = int(input("Enter the number of clusters: "))
    num_points = int(input("Enter the number of data points: "))

    data = generate_random_data(num_points, num_clusters)
    write_data_to_file(data)

if __name__ == "__main__":
    main()
