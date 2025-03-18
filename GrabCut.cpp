#include "GrabCut.hpp"
#include <tuple>

cv::Mat MultiScaleCanny(const cv::Mat& image) {
    cv::Mat edges_combined = cv::Mat::zeros(image.size(), CV_8U);
    std::vector<double> sigma_list = {1.0, 2.0, 3.0};

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

cv::Mat SimpleGrabCutForeground(const cv::Mat& image) {
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray_image, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    mask.convertTo(mask, CV_64F);

    return mask;
}

// Native function that will be called from .NET
extern "C" __declspec(dllexport) void ComputeForegroundProbability(const char* imagePath, double* foregroundScore, double* backgroundScore) {
    // Read the image using OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        return; // If the image is empty, just return
    }

    // Convert to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Apply Canny edge detection
    cv::Mat edges_refined = MultiScaleCanny(gray_image);

    // Compute foreground probability using GrabCut algorithm
    cv::Mat fg_prob = SimpleGrabCutForeground(image);
    fg_prob = (fg_prob > 0);
    fg_prob.convertTo(fg_prob, CV_64F);

    // Calculate the foreground and background scores
    double minVal, max_prob;
    cv::minMaxLoc(fg_prob, &minVal, &max_prob);
    if (max_prob > 0) {
        fg_prob /= max_prob;
    }

    *foregroundScore = cv::mean(fg_prob)[0];
    *backgroundScore = 1.0 - (*foregroundScore);
}
