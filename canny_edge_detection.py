import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from sklearn.mixture import GaussianMixture

# Simple Canny Edge Detection Algorithm
def canny_edge_detector(image):
    smoothed_image = gaussian_filter(image, sigma=1.4)
    dx = sobel(smoothed_image, axis=1)
    dy = sobel(smoothed_image, axis=0)
    gradient_magnitude = np.hypot(dx, dy)
    gradient_direction = np.arctan2(dy, dx)

    suppressed_image = np.zeros_like(gradient_magnitude)
    angle = np.degrees(gradient_direction) % 180
    
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if (angle[i, j] < 22.5) or (angle[i, j] >= 157.5):
                p, r = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]
            elif (angle[i, j] < 67.5):
                p, r = gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]
            elif (angle[i, j] < 112.5):
                p, r = gradient_magnitude[i+1, j], gradient_magnitude[i-1, j]
            else:
                p, r = gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]
            
            if gradient_magnitude[i, j] >= p and gradient_magnitude[i, j] >= r:
                suppressed_image[i, j] = gradient_magnitude[i, j]
    
    high_threshold = np.max(suppressed_image) * 0.2
    low_threshold = high_threshold * 0.1
    strong_edges = (suppressed_image > high_threshold)
    weak_edges = (suppressed_image >= low_threshold) & (suppressed_image <= high_threshold)
    
    final_edges = np.zeros_like(suppressed_image)
    for i in range(1, suppressed_image.shape[0] - 1):
        for j in range(1, suppressed_image.shape[1] - 1):
            if strong_edges[i, j] or (weak_edges[i, j] and np.any(strong_edges[i-1:i+2, j-1:j+2])):
                final_edges[i, j] = 255
    
    return final_edges

# Function to Compute Foreground Probability using Gaussian Mixture Model
def foreground_probability(gray_image, n_components=2):
    pixels = gray_image.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(pixels)
    prob_foreground = gmm.predict_proba(pixels)[:, 1]
    return prob_foreground.reshape(gray_image.shape)

# Load Image
image = plt.imread('landscape_4k.jpg')
gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Apply Canny Edge Detection
edges = canny_edge_detector(gray_image)

# Compute Foreground Probability
fg_prob = foreground_probability(gray_image)

# Display Results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(edges, cmap='gray')
ax[0].set_title("Canny Edges")
ax[1].imshow(fg_prob, cmap='jet')
ax[1].set_title("Foreground Probability (GMM)")
ax[2].imshow(edges * fg_prob, cmap='jet')
ax[2].set_title("Final Combined Probability")
plt.show()