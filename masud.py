import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.style.use('classic')

# Load the image
img = Image.open('/content/drive/MyDrive/Colab Notebooks/test.png')

# Convert the image to grayscale
imggray = img.convert('L')

# Rotate the image
imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.rot90(imgmat, k=1)

# Display the original image
plt.figure(figsize=(9, 6))
plt.imshow(imgmat, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

def pca_reconstruction(imggray, vals):
    imgmat = np.array(list(imggray.getdata(band=0)), float)
    imgmat.shape = (imggray.size[1], imggray.size[0])
    cov_mat = imgmat - np.mean(imgmat, axis=1)[:, np.newaxis]
    eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))
    p = np.size(eig_vec, axis=1)
    idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, idx]
    eig_val = eig_val[idx]
    if vals < p or vals > 0:
        eig_vec = eig_vec[:, :vals]
    score = np.dot(eig_vec.T, cov_mat)
    recon = np.dot(eig_vec, score) + np.mean(imgmat, axis=1)[:, np.newaxis]
    return recon

# Perform PCA reconstruction with 200 components
Lambda1 = pca_reconstruction(imggray, 200)
print("Original Image Shape:", imgmat.shape)
print("Reconstructed Image Shape (200 Components):", Lambda1.shape)

# Perform PCA reconstruction with 50 components
Lambda2 = pca_reconstruction(imggray, 50)
print("Original Image Shape:", imgmat.shape)
print("Reconstructed Image Shape (50 Components):", Lambda2.shape)

# Perform PCA reconstruction with 10 components
Lambda3 = pca_reconstruction(imggray, 10)
print("Original Image Shape:", imgmat.shape)
print("Reconstructed Image Shape (10 Components):", Lambda3.shape)

# Display the reconstructed images
fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4)
ax1.axis('off')
ax1.imshow(imggray, cmap=plt.get_cmap('gray'))
ax1.set_title('True Image')
ax2.axis('off')
ax2.imshow(Lambda1, cmap=plt.get_cmap('gray'))
ax2.set_title('200 Components')
ax3.axis('off')
ax3.imshow(Lambda2, cmap=plt.get_cmap('gray'))
ax3.set_title('50 Components')
ax4.axis('off')
ax4.imshow(Lambda3, cmap=plt.get_cmap('gray'))
ax4.set_title('10 Components')
plt.show()