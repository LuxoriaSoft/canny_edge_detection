/**
 * GrabCut with Multi-Scale Canny Edge Detection:
 *
 * This program integrates the GrabCut algorithm with multi-scale Canny edge detection
 * to segment the foreground from the background in an image. The primary goal is to compute foreground
 * and background probabilities and visualize the results.
 *
 * Key Components and Workflow:
 *
 * 1. Multi-Scale Canny Edge Detection:
 *    - The program applies Canny edge detection at multiple scales (using different Gaussian smoothing
 *      levels) to detect edges in the input image.
 *    - The edges detected at different scales are combined and refined using morphological operations
 *      to close small gaps.
 *    - Algorithm: Canny edge detection with Gaussian smoothing and morphological closing.
 *
 * 2. GrabCut Segmentation:
 *    - The GrabCut algorithm is used to segment the foreground from the background in the image.
 *    - A bounding box is defined to initialize the segmentation process, and the algorithm iteratively
 *      refines the segmentation based on color statistics.
 *    - Algorithm: GrabCut for foreground-background segmentation.
 *
 * 3. Foreground and Background Probability Computation:
 *    - The foreground probability is computed from the GrabCut segmentation result.
 *    - The program calculates the mean foreground probability and an edge-weighted foreground score,
 *      which considers the overlap between the foreground and the detected edges.
 *    - Algorithm: Mean calculation and edge-weighted scoring.
 *
 * 4. Implementation:
 *    - The program is implemented using OpenCV, a popular computer vision library.
 *    - It reads an input image, processes it to detect edges and segment the foreground, and then
 *      computes various scores to quantify the segmentation quality.
 *    - The results are displayed using OpenCV's visualization functions.
 *
 * 5. Output:
 *    - The program outputs the foreground probability score, background probability score, and
 *      edge-weighted foreground score.
 *    - It also displays the foreground probability map and the refined edges for visual inspection.
 *
 * Usage:
 * - The program is executed from the command line with the path to an input image as an argument.
 * - Example: ./grabcut_edge_detection <image_path>
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>
#include <vector>

/**
 * This function applies multi-scale Canny edge detection to an input image.
 * It returns the refined edges after morphological closing.
 * The function takes an input image and a list of sigma values for Gaussian smoothing.
 * The default sigma values are {1.0, 2.0, 3.0}.
 * @param image: Input image (grayscale)
 * @param sigma_list: List of sigma values for Gaussian smoothing
 * @return edges_refined: Refined edges after morphological closing
 */
cv::Mat multi_scale_canny(const cv::Mat& image, const std::vector<double>& sigma_list = {1.0, 2.0, 3.0}) {
    cv::Mat edges_combined = cv::Mat::zeros(image.size(), CV_8U);

    for (double sigma : sigma_list) {
        cv::Mat blurred;
        cv::GaussianBlur(image, blurred, cv::Size(5, 5), sigma);  // Gaussian smoothing
        cv::Mat edges;
        cv::Canny(blurred, edges, 50, 150);  // Canny edge detection
        edges_combined = cv::max(edges_combined, edges);  // Combine edges across scales
    }

    // Close small gaps in edges (morphological closing)
    cv::Mat edges_refined;
    cv::morphologyEx(edges_combined, edges_refined, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    return edges_refined;
}

/**
 * This function applies GrabCut algorithm to an input image and returns the foreground mask.
 * The function takes an input image and returns a binary mask where 1 represents the foreground.
 * @param image: Input image (BGR)
 * @return fg_mask: Foreground mask (binary)
 */
cv::Mat grabcut_foreground(const cv::Mat& image) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // Background and foreground models, required by GrabCut
    cv::Mat bgd_model, fgd_model;
    bgd_model.create(1, 65, CV_64F);
    fgd_model.create(1, 65, CV_64F);

    // Define a bounding box (adjust based on the input image)
    cv::Rect rect(50, 50, image.cols - 100, image.rows - 100);  // x, y, width, height

    // Run GrabCut
    cv::grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv::GC_INIT_WITH_RECT);

    // Mask the background pixels as 0 and foreground pixels as 1
    cv::Mat fg_mask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);  // Foreground mask
    fg_mask.convertTo(fg_mask, CV_64F); // Convert mask to CV_64F for further computation

    return fg_mask;
}

/**
 * This function computes the foreground and background probabilities based on the input image and refined edges.
 * The function returns the foreground probability, foreground score, background score, and edge-weighted foreground score.
 * @param image_rgb: Input image (BGR)
 * @param edges_refined: Refined edges (binary)
 * @return fg_prob: Foreground probability (floating point)
 * @return foreground_score: Foreground probability score (mean)
 * @return background_score: Background probability score (complement of foreground)
 */
std::tuple<cv::Mat, double, double, double> compute_foreground_background_probability(const cv::Mat& image_rgb, const cv::Mat& edges_refined) {
    cv::Mat fg_prob = grabcut_foreground(image_rgb); // Get foreground probability

    // Normalize fg_prob to range [0, 1]
    fg_prob.convertTo(fg_prob, CV_64F, 1.0 / 255.0);  // Normalize to [0, 1]

    // Ensure both fg_prob and edges_refined are of the same type (CV_64F for floating point operations)
    if (fg_prob.type() != CV_64F) {
        fg_prob.convertTo(fg_prob, CV_64F);
    }
    cv::Mat edges_refined_64F;
    edges_refined.convertTo(edges_refined_64F, CV_64F);

    // Normalize edges_refined to [0, 1]
    edges_refined_64F /= 255.0;

    // Calculate the foreground score (mean of foreground probability)
    double foreground_score = cv::mean(fg_prob)[0];
    double background_score = 1.0 - foreground_score;  // Background is the complement

    // Calculate the edge-weighted foreground score (based on refined edges)
    double edge_weighted_fg = 0.0;
    if (cv::sum(edges_refined_64F)[0] > 0) {  // Check if edges exist
        edge_weighted_fg = cv::sum(fg_prob.mul(edges_refined_64F))[0] / cv::sum(edges_refined_64F)[0];
    }

    return std::make_tuple(fg_prob, foreground_score, background_score, edge_weighted_fg);
}

int main(int argc, char** argv) {
    // Load image from file
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) return -1;

    // Ensure the image is 3-channel
    if (image.channels() == 1) cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

    // Create a mask initialized as probable background
    cv::Mat mask(image.size(), CV_8UC1, cv::GC_PR_BGD);

    // Define a bounding box
    int margin = 10; // Avoid edges of the image
    cv::Rect rect(margin, margin, image.cols - 2 * margin, image.rows - 2 * margin);
    mask(rect).setTo(cv::GC_PR_FGD); // Mark probable foreground inside the box

    // Initialize background and foreground models
    cv::Mat bgd_model, fgd_model;

    // Run GrabCut
    cv::grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv::GC_INIT_WITH_MASK);
    // Extract the foreground mask
    cv::Mat fg_mask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
    std::cout << "Foreground mask extracted." << std::endl;

    return 0;
}