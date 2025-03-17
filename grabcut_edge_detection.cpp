/**
 * GrabCut with Edge Detection
 * This program returns the foreground/background probabilities of an image using GrabCut and edge detection.
 * The score is gapped between 0 and 1, where 0 indicates the background and 1 indicates the foreground.
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>
#include <vector>

/**
 * @brief Apply multi-scale Canny edge detection to an image.
 * @param image The input image.
 * @param sigma_list A list of Gaussian kernel standard deviations for blurring.
 * @return The refined edges image.
 */
cv::Mat multi_scale_canny(const cv::Mat& image, const std::vector<double>& sigma_list = {1.0, 2.0, 3.0}) {
    cv::Mat edges_combined = cv::Mat::zeros(image.size(), CV_8U);

    for (double sigma : sigma_list) {
        cv::Mat blurred;
        int kernel_size = std::max(3, static_cast<int>(std::round(sigma * 2.0 + 1)));
        cv::GaussianBlur(image, blurred, cv::Size(kernel_size, kernel_size), sigma);
        cv::Mat edges;
        cv::Canny(blurred, edges, 50, 150);
        edges_combined = cv::max(edges_combined, edges);
    }

    cv::Mat edges_refined;
    cv::morphologyEx(edges_combined, edges_refined, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    return edges_refined;
}

/**
 * @brief Apply a simple GrabCut algorithm to an image to extract the foreground.
 * This function converts the input image to grayscale, applies Otsu's thresholding
 * to create a binary mask, and then converts the mask to a double precision matrix.
 * @param image The input image.
 * @return The binary mask of the foreground.
 */
cv::Mat simple_grabcut_foreground(const cv::Mat& image) {
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Apply a simple threshold to create an initial binary mask
    cv::Mat mask;
    cv::threshold(gray_image, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Convert the mask to double for further processing
    mask.convertTo(mask, CV_64F);

    return mask;
}

/**
 * @brief Apply an upgraded GrabCut algorithm to an image to extract the foreground.
 * @param image The input image.
 * @return The binary mask of the foreground.
 */
cv::Mat upgraded_grabcut_foreground(const cv::Mat& image) {
    // Apply bilateral filtering to smooth the image while preserving edges
    cv::Mat filtered_image;
    cv::bilateralFilter(image, filtered_image, 9, 75, 75);

    // Convert the image to the Lab color space for better color segmentation
    cv::Mat lab_image;
    cv::cvtColor(filtered_image, lab_image, cv::COLOR_BGR2Lab);

    // Split the Lab image into L, a, and b channels
    std::vector<cv::Mat> lab_channels(3);
    cv::split(lab_image, lab_channels);

    // Apply Otsu's thresholding on the a-channel (which often captures color edges well)
    cv::Mat mask;
    cv::threshold(lab_channels[1], mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Apply morphological operations to clean up the mask
    cv::Mat morph_mask;
    cv::morphologyEx(mask, morph_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::morphologyEx(morph_mask, morph_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    // Convert the mask to double for further processing
    morph_mask.convertTo(morph_mask, CV_64F);

    return morph_mask;
}

/**
 * @brief Compute the foreground and background probabilities of an image using GrabCut and edge detection.
 * @param image_rgb The input image in RGB color space.
 * @param edges_refined The refined edges image.
 * @return A tuple containing the foreground probability mask, foreground score, background score, and edge-weighted foreground score.
 */
std::tuple<cv::Mat, double, double, double> compute_foreground_background_probability(
    const cv::Mat& image_rgb, const cv::Mat& edges_refined) {

    std::cout << "Computing grabcut mask... ";
    cv::Mat fg_prob = simple_grabcut_foreground(image_rgb);
    std::cout << "done!" << std::endl;

    // Ensure binary mask before normalization
    fg_prob = (fg_prob > 0);
    fg_prob.convertTo(fg_prob, CV_64F);

    // Use minMaxLoc to find the actual max value in the matrix
    double minVal, max_prob;
    cv::minMaxLoc(fg_prob, &minVal, &max_prob);

    if (max_prob > 0) {
        fg_prob /= max_prob;  // Normalize to range [0,1]
    }

    double foreground_score = cv::mean(fg_prob)[0];
    foreground_score = std::clamp(foreground_score, 0.0, 1.0);
    double background_score = 1.0 - foreground_score;

    cv::Mat edges_refined_64F;
    edges_refined.convertTo(edges_refined_64F, CV_64F);
    edges_refined_64F /= 255.0;

    double edge_weighted_fg = 0.0;
    if (cv::sum(edges_refined_64F)[0] > 0) {
        edge_weighted_fg = cv::sum(fg_prob.mul(edges_refined_64F))[0] /
                           std::max(1.0, cv::sum(edges_refined_64F)[0]);
    }

    return std::make_tuple(fg_prob, foreground_score, background_score, edge_weighted_fg);
}

/**
 * @brief Main function to demonstrate GrabCut with edge detection.
 * @param ac The number of command-line arguments.
 * @param av The command-line arguments.
 * @return The exit status.
 */
int main(int ac, char** av) {
    if (ac != 2) {
        std::cout << "Usage: ./grabcut_edge_detection <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(av[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    if (image.cols < 100 || image.rows < 100) {
        std::cout << "Resizing image to avoid failure with GrabCut..." << std::endl;
        cv::resize(image, image, cv::Size(100, 100));
    }

    std::cout << "Loaded image with size: " << image.cols << " x " << image.rows << std::endl;

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    std::cout << "Applying multi-scale Canny edge detection..." << std::endl;
    cv::Mat edges_refined = multi_scale_canny(gray_image);

    std::cout << "Computing foreground and background probabilities..." << std::endl;
    cv::Mat fg_prob;
    double foreground_score, background_score, edge_weighted_fg;
    std::tie(fg_prob, foreground_score, background_score, edge_weighted_fg) =
        compute_foreground_background_probability(image, edges_refined);

    std::cout << "Foreground Probability Score: " << foreground_score << std::endl;
    std::cout << "Background Probability Score: " << background_score << std::endl;
    std::cout << "Edge-Weighted Foreground Score: " << edge_weighted_fg << std::endl;

    return 0;
}
