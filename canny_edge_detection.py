import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel

# Simple Canny Edge Detection Algorithm
# Sources :
# https://en.wikipedia.org/wiki/Canny_edge_detector
# https://medium.com/@rohit-krishna/coding-canny-edge-detection-algorithm-from-scratch-in-python-232e1fdceac7
# https://medium.com/@abhisheksriram845/canny-edge-detection-explained-and-compared-with-opencv-in-python-57a161b4bd19

def canny_edge_detector(image):
    # Apply Noise Reduction Filter
    smoothed_image = gaussian_filter(image, sigma=1.4)

    # Compute Gradient Calculation
    dx = sobel(smoothed_image, axis=1)
    dy = sobel(smoothed_image, axis=0)
    gradient_magnitude = np.hypot(dx, dy)
    gradient_direction = np.arctan2(dy, dx)

    # Apply Non-Maximum Suppression Algorithm
    suppressed_image = np.zeros_like(gradient_magnitude)
    angle = np.degrees(gradient_direction) % 180
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if (angle[i, j] >= 0 and angle[i, j] < 22.5) or (angle[i, j] >= 157.5 and angle[i, j] <= 180):
                p, r = gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]
            elif (angle[i, j] >= 22.5 and angle[i, j] < 67.5):
                p, r = gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]
            elif (angle[i, j] >= 67.5 and angle[i, j] < 112.5):
                p, r = gradient_magnitude[i+1, j], gradient_magnitude[i-1, j]
            else:
                p, r = gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]

            if gradient_magnitude[i, j] >= p and gradient_magnitude[i, j] >= r:
                suppressed_image[i, j] = gradient_magnitude[i, j]

    # Apply Double Thresholding
    high_threshold = np.max(suppressed_image) * 0.15
    low_threshold = high_threshold * 0.05
    strong_edges = (suppressed_image > high_threshold)
    weak_edges = (suppressed_image >= low_threshold) & (suppressed_image <= high_threshold)

    # Edge Tracking by Hysteresis
    final_edges = np.zeros_like(suppressed_image)
    for i in range(1, suppressed_image.shape[0] - 1):
        for j in range(1, suppressed_image.shape[1] - 1):
            if strong_edges[i, j]:
                final_edges[i, j] = 255
            elif weak_edges[i, j]:
                if (strong_edges[i+1, j] or strong_edges[i-1, j] or
                    strong_edges[i, j+1] or strong_edges[i, j-1] or
                    strong_edges[i+1, j+1] or strong_edges[i+1, j-1] or
                    strong_edges[i-1, j+1] or strong_edges[i-1, j-1]):
                    final_edges[i, j] = 255

    return final_edges

# Load Image
image = plt.imread('image.png')
# Convert it to Grayscale
gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
# Apply Canny Edge Detection
edges = canny_edge_detector(gray_image)

# Show the result
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.show()
