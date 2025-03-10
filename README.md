
# GrabCut with Multi-Scale Canny Edge Detection

## Overview

This project demonstrates the integration of the **GrabCut segmentation** algorithm with **multi-scale Canny edge detection** to segment the foreground from an image and refine the result using edge information. The combined approach results in more accurate foreground extraction, particularly in images with complex edges. Additionally, the system computes **foreground probability**, **background probability**, and **edge-weighted foreground scores** to provide insights into the segmentation quality.

This code is provided in both **Python** and **C++**, allowing for cross-platform usage. The Python version uses libraries like **OpenCV**, **NumPy**, and **Matplotlib**, while the C++ version is built around **OpenCV**.

## Features

- **GrabCut Segmentation**: A background subtraction method for segmenting an object from its background.
- **Multi-Scale Canny Edge Detection**: Combines results from different levels of Gaussian smoothing to enhance edge detection.
- **Foreground and Background Probability Calculation**: Computes the likelihood of a pixel being part of the foreground or background.
- **Edge-Weighted Foreground Score**: Evaluates the quality of the foreground by considering edges to refine the segmentation.

## Requirements

### Python Version

- **Python** (v3.x or higher)
- **OpenCV**: For image processing, segmentation, and edge detection.
- **NumPy**: For array manipulations and numerical operations.
- **Matplotlib**: For visualizing results.
- **Scikit-Image**: For morphological operations.

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### C++ Version

- **C++ Compiler** (e.g., GCC or Clang)
- **OpenCV** (make sure OpenCV is installed and properly configured for C++)

To install OpenCV on a Linux system:

```bash
sudo apt-get install libopencv-dev
```

## How It Works

### Python Version

1. **GrabCut Segmentation**:
    - The **GrabCut algorithm** is used to segment the foreground of an image. It is initialized with a bounding box, and an iterative algorithm determines the foreground and background models.

2. **Multi-Scale Canny Edge Detection**:
    - Multi-scale **Canny edge detection** is applied by using Gaussian blur with different levels of smoothing (sigma values). The results from these scales are combined to detect edges at different resolutions.
    - The edge detection result is then refined using **morphological closing** to remove small gaps in edges.

3. **Foreground and Background Probability Calculation**:
    - The **foreground probability** is the mean value of the foreground mask obtained from the GrabCut segmentation.
    - The **background probability** is simply the complement of the foreground probability.
    - An **edge-weighted foreground score** is calculated by multiplying the foreground probability map with the edge map and averaging the result. This emphasizes areas where edges are more significant.

4. **Visualization**:
    - The results, including the foreground probability map, refined edges, and a combination of foreground and edges, are displayed using **Matplotlib**.

#### Python Code:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import closing, footprint_rectangle

def grabcut_foreground(image):
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1]-100, image.shape[0]-100)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float64)
    return fg_mask

def multi_scale_canny(image, sigma_list=[1.0, 2.0, 3.0]):
    edges_combined = np.zeros_like(image)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(image, (5, 5), sigma)
        edges = cv2.Canny((blurred * 255).astype(np.uint8), 50, 150)
        edges_combined = np.maximum(edges_combined, edges)
    edges_refined = closing(edges_combined, footprint_rectangle((3, 3)))
    return edges_refined

def compute_foreground_background_probability(image_rgb, edges_refined):
    fg_prob = grabcut_foreground(image_rgb)
    foreground_score = np.mean(fg_prob)
    background_score = 1 - foreground_score
    edge_weighted_fg = np.sum(fg_prob * (edges_refined / 255)) / np.sum(edges_refined / 255) if np.sum(edges_refined) > 0 else 0
    return fg_prob, foreground_score, background_score, edge_weighted_fg
```

### C++ Version

The C++ version implements the same core steps:

1. **Multi-Scale Canny Edge Detection**: Applies Canny edge detection at multiple smoothing scales.
2. **GrabCut Segmentation**: Uses the **GrabCut** algorithm to separate the foreground from the background.
3. **Foreground/Background Probability Calculation**: Calculates foreground and background probabilities based on the segmentation.
4. **Visualization**: Displays the results using OpenCV's `imshow()`.

#### C++ Code:

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat multi_scale_canny(const Mat& image, const vector<double>& sigma_list = {1.0, 2.0, 3.0}) {
    Mat edges_combined = Mat::zeros(image.size(), CV_8U);
    for (double sigma : sigma_list) {
        Mat blurred;
        GaussianBlur(image, blurred, Size(5, 5), sigma);
        Mat edges;
        Canny(blurred, edges, 50, 150);
        edges_combined = max(edges_combined, edges);
    }
    Mat edges_refined;
    morphologyEx(edges_combined, edges_refined, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    return edges_refined;
}

Mat grabcut_foreground(const Mat& image) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    Mat bgd_model, fgd_model;
    bgd_model.create(1, 65, CV_64F);
    fgd_model.create(1, 65, CV_64F);
    Rect rect(50, 50, image.cols - 100, image.rows - 100);
    grabCut(image, mask, rect, bgd_model, fgd_model, 5, GC_INIT_WITH_RECT);
    Mat fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD);
    fg_mask.convertTo(fg_mask, CV_64F);
    return fg_mask;
}

tuple<Mat, double, double, double> compute_foreground_background_probability(const Mat& image_rgb, const Mat& edges_refined) {
    Mat fg_prob = grabcut_foreground(image_rgb);
    fg_prob.convertTo(fg_prob, CV_64F, 1.0 / 255.0);
    Mat edges_refined_64F;
    edges_refined.convertTo(edges_refined_64F, CV_64F);
    edges_refined_64F /= 255.0;

    double foreground_score = mean(fg_prob)[0];
    double background_score = 1.0 - foreground_score;
    double edge_weighted_fg = 0.0;
    if (sum(edges_refined_64F)[0] > 0) {
        edge_weighted_fg = sum(fg_prob.mul(edges_refined_64F))[0] / sum(edges_refined_64F)[0];
    }

    return make_tuple(fg_prob, foreground_score, background_score, edge_weighted_fg);
}
```

## Results

The system will display the following:

- **Foreground Probability Map**: This map visualizes the probability that each pixel is part of the foreground.
- **Refined Edges**: The edges of the image after multi-scale Canny edge detection and morphological closing.
- **Combined Foreground and Edges**: This result highlights the foreground using both GrabCut segmentation and edge information.

### Example Output:

The program will print the following scores:

- **Foreground Probability Score**: The average foreground probability value.
- **Background Probability Score**: The complement of the foreground probability.
- **Edge-Weighted Foreground Score**: The average foreground score weighted by edge information.

## Use Case

This project is useful in scenarios such as:

- **Image segmentation** where accurate foreground extraction is required.
- **Computer Vision applications** that need to separate objects from complex backgrounds.
- **Image Processing pipelines** that enhance the segmentation quality by considering edges.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
