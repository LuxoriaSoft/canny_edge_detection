import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, footprint_rectangle

# Function for GrabCut segmentation
def grabcut_foreground(image):
    # Convert image to 8-bit color
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image

    # Initialize a mask for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Background and foreground models, required by GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define a bounding box (change as per your requirement)
    rect = (50, 50, image.shape[1]-100, image.shape[0]-100)  # (x, y, width, height)
    
    # Run GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Mask the background pixels as 0 and foreground pixels as 1
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float64)
    return fg_mask

# Function to refine with Canny Edge Detection
def multi_scale_canny(image, sigma_list=[1.0, 2.0, 3.0]):
    edges_combined = np.zeros_like(image)
    
    # Apply multi-scale Canny edge detection
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(image, (5, 5), sigma)  # Gaussian smoothing
        edges = cv2.Canny((blurred * 255).astype(np.uint8), 50, 150)
        edges_combined = np.maximum(edges_combined, edges)  # Combine scales
    
    # Close small gaps in edges
    edges_refined = closing(edges_combined, footprint_rectangle((3, 3)))  # Close gaps
    return edges_refined

# Function to compute foreground and background probabilities
def compute_foreground_background_probability(image_rgb, edges_refined):
    fg_prob = grabcut_foreground(image_rgb)  # Extract foreground probability map

    # Calculate the foreground score (mean of foreground probability)
    foreground_score = np.mean(fg_prob)
    background_score = 1 - foreground_score  # Background is the complement
    
    # Calculate the edge-weighted foreground score (based on refined edges)
    edge_weighted_fg = np.sum(fg_prob * (edges_refined / 255)) / np.sum(edges_refined / 255) if np.sum(edges_refined) > 0 else 0
    
    return fg_prob, foreground_score, background_score, edge_weighted_fg

# Load and preprocess the image
image = cv2.imread("image2.jpg")  # Load your image here (replace with your file path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0  # Normalize to [0,1]

# Apply multi-scale Canny for edge detection
edges_refined = multi_scale_canny(gray_image)

# Compute foreground and background probabilities
fg_prob, foreground_score, background_score, edge_weighted_fg = compute_foreground_background_probability(image_rgb, edges_refined)

# Display results using matplotlib
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(fg_prob, cmap='jet')
ax[0].set_title("GrabCut Foreground Probability")

ax[1].imshow(edges_refined, cmap='gray')
ax[1].set_title("Canny Edge Detection")

ax[2].imshow(fg_prob * edges_refined, cmap='jet')
ax[2].set_title("Combined Foreground Probability and Edges")

plt.show()

# Print out the scores
print(f"Foreground Probability Score: {foreground_score:.4f}")
print(f"Background Probability Score: {background_score:.4f}")
print(f"Edge-Weighted Foreground Score: {edge_weighted_fg:.4f}")
