import cv2
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Compute the mean of the input data
        self.mean = np.mean(X, axis=0)

        # Center the data by subtracting the mean
        X_centered = X - self.mean

        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

    def fit_transform(self, X):
        # Center the data by subtracting the mean
        X_centered = X - self.mean

        # Project the data onto the principal components
        transformed_data = np.dot(X_centered, self.components)

        return transformed_data

    def inverse_transform(self, X_transformed):
        # Project the data back to the original space
        X_reconstructed = np.dot(X_transformed, self.components.T)

        # Add back the mean to the reconstructed data
        X_original = X_reconstructed + self.mean

        return X_original


def image_compressor(img):
    red, green, blue = cv2.split(img)
    pca = PCA(n_components=10)

    pca.fit(red)
    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)

    pca.fit(green)
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)

    pca.fit(blue)
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)
    img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)
    return img_compressed


def calculate_distance(image1, image2):
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape.")

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Flatten the image arrays
    image1_flattened = image1_array.reshape(-1, image1_array.shape[-1])
    image2_flattened = image2_array.reshape(-1, image2_array.shape[-1])

    # Calculate the distance matrix
    distance_matrix = cdist(image1_flattened, image2_flattened, metric='euclidean')

    return distance_matrix


img1 = cv2.cvtColor(cv2.imread('data/image.jpg'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('data/image2.jpg'), cv2.COLOR_BGR2RGB)

img1_compressed = image_compressor(img1)
img2_compressed = image_compressor(img2)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axes[0, 0].imshow(img1)
axes[0, 0].set_title('Image 1')
axes[0, 1].imshow(img2)
axes[0, 1].set_title('Image 2')
axes[1, 0].imshow(img1_compressed)
axes[1, 0].set_title('Image 1 Compressed')
axes[1, 1].imshow(img2_compressed)
axes[1, 1].set_title('Image 2 Compressed')
plt.show()

img1_compressed.resize((100, 100))
img2_compressed.resize((100, 100))
distance_matrix = calculate_distance(img1_compressed, img2_compressed)

plt.imshow(img1_compressed)
plt.show()
