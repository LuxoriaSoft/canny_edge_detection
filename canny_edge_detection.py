import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, closing, footprint_rectangle
from scipy.special import softmax

# Multi-Scale Canny Edge Detection with Morphological Refinement
def multi_scale_canny(image, sigma_list=[1.0, 2.0, 3.0], low_threshold=50, high_threshold=150):
    edges_combined = np.zeros_like(image)
    
    for sigma in sigma_list:
        blurred = gaussian_filter(image, sigma=sigma)  # Multi-scale smoothing
        edges = cv2.Canny((blurred * 255).astype(np.uint8), low_threshold, high_threshold)
        edges_combined = np.maximum(edges_combined, edges)  # Combine scales
    
    # Apply Morphological Refinement
    edges_refined = closing(edges_combined, footprint_rectangle((3, 3)))  # Close gaps
    edges_refined = dilation(edges_refined, footprint_rectangle((2, 2)))  # Strengthen edges
    return edges_refined

# GrabCut Foreground Segmentation with Improved Mask Refinement
def grabcut_foreground(image_color):
    # Ensure image is in CV_8UC3 format
    image_color = (image_color * 255).astype(np.uint8) if image_color.max() <= 1 else image_color

    # Create an initial mask, background and foreground models
    mask = np.zeros(image_color.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    
    # Define a bounding box for GrabCut (dynamically adjust if needed)
    rect = (50, 50, image_color.shape[1]-50, image_color.shape[0]-50)  # Adjust the bounding box as needed
    cv2.grabCut(image_color, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float64)  # Extract foreground
    fg_mask = closing(fg_mask, footprint_rectangle((5, 5)))  # Apply closing to refine the mask
    return fg_mask  # Foreground probability map

# Combine Canny Edges and Foreground Information
def combine_edges_foreground(edges, fg_prob, alpha=0.7):
    # Normalize the foreground probability map to [0, 1]
    fg_prob_norm = (fg_prob - np.min(fg_prob)) / (np.max(fg_prob) - np.min(fg_prob))
    
    # Apply a weighted sum (alpha controls the balance between edges and foreground)
    combined = alpha * edges + (1 - alpha) * fg_prob_norm
    return np.clip(combined, 0, 1)  # Ensure the combined result is within [0, 1]

# Apply Thresholding and Further Refinement
def threshold_and_refine(combined, threshold=0.5):
    # Threshold the combined result to create a binary mask
    binary_mask = (combined >= threshold).astype(np.uint8)
    
    # Apply additional morphological operations to clean up the binary mask
    refined_mask = closing(binary_mask, footprint_rectangle((5, 5)))  # Closing to fill holes
    refined_mask = dilation(refined_mask, footprint_rectangle((3, 3)))  # Strengthen mask
    
    return refined_mask

# Compute Probability Scores
def compute_probability_scores(fg_prob, edges):
    fg_prob_norm = (fg_prob - np.min(fg_prob)) / (np.max(fg_prob) - np.min(fg_prob))  # Normalize
    weighted_fg = softmax(fg_prob_norm.ravel())  # Apply softmax

    foreground_score = np.mean(weighted_fg)
    edge_weighted_score = np.sum(fg_prob * (edges / 255)) / np.sum(edges / 255)
    
    return foreground_score, edge_weighted_score

# Load and Preprocess Image
image = cv2.imread("image2.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (for correct display)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0  # Normalize to [0,1]

# Apply Multi-Scale Canny
edges = multi_scale_canny(gray_image)

# Compute Foreground Probability using GrabCut
fg_prob = grabcut_foreground(image_rgb)  # Use RGB image instead of grayscale

# Combine Edge and Foreground Information
combined = combine_edges_foreground(edges, fg_prob, alpha=0.7)  # Adjust alpha as needed

# Apply Thresholding to refine the result further
refined_mask = threshold_and_refine(combined, threshold=0.5)

# Compute Scores
foreground_score, edge_weighted_score = compute_probability_scores(fg_prob, edges)

# Display Results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(edges, cmap='gray')
ax[0].set_title("Multi-Scale Canny Edges")

ax[1].imshow(fg_prob, cmap='jet')
ax[1].set_title("GrabCut Foreground Probability")

ax[2].imshow(refined_mask, cmap='gray')
ax[2].set_title("Final Refined Segmentation")

plt.show()

# Print Probability Scores
print(f"Foreground Probability Score: {foreground_score:.4f}")
print(f"Edge-Weighted Foreground Score: {edge_weighted_score:.4f}")
