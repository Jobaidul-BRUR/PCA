import numpy as np
from sklearn.datasets import fetch_lfw_people

def pca(data, num_components):
    # Normalize the data
    data_mean = np.mean(data, axis=0)
    data_normalized = data - data_mean

    # Calculate the covariance matrix
    covariance_matrix = np.cov(data_normalized, rowvar=False)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Project the data onto the selected eigenvectors
    projected_data = np.dot(data_normalized, selected_eigenvectors)

    return projected_data

def calculate_distance_matrix(data):
    num_samples = data.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            distance = np.linalg.norm(data[i] - data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

# Load the LFW dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=70)

# Extract the data and target labels from the dataset
data = lfw_dataset.data

# Perform PCA with 2 components
num_components = 2
projected_data = pca(data, num_components)

# Calculate the distance matrix for the projected data
distance_matrix = calculate_distance_matrix(projected_data)

print("Distance matrix:")
print(distance_matrix)