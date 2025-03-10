#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <tuple>

using namespace cv;
using namespace std;

// Function to apply multi-scale Canny edge detection
Mat multi_scale_canny(const Mat& image, const vector<double>& sigma_list = {1.0, 2.0, 3.0}) {
    Mat edges_combined = Mat::zeros(image.size(), CV_8U);

    for (double sigma : sigma_list) {
        Mat blurred;
        GaussianBlur(image, blurred, Size(5, 5), sigma);  // Gaussian smoothing
        Mat edges;
        Canny(blurred, edges, 50, 150);  // Canny edge detection
        edges_combined = max(edges_combined, edges);  // Combine edges across scales
    }

    // Close small gaps in edges (morphological closing)
    Mat edges_refined;
    morphologyEx(edges_combined, edges_refined, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));

    return edges_refined;
}

// Function to apply GrabCut segmentation
Mat grabcut_foreground(const Mat& image) {
    Mat mask = Mat::zeros(image.size(), CV_8UC1);

    // Background and foreground models, required by GrabCut
    Mat bgd_model, fgd_model;
    bgd_model.create(1, 65, CV_64F);
    fgd_model.create(1, 65, CV_64F);

    // Define a bounding box (adjust based on the input image)
    Rect rect(50, 50, image.cols - 100, image.rows - 100);  // x, y, width, height

    // Run GrabCut
    grabCut(image, mask, rect, bgd_model, fgd_model, 5, GC_INIT_WITH_RECT);

    // Mask the background pixels as 0 and foreground pixels as 1
    Mat fg_mask = (mask == GC_FGD) | (mask == GC_PR_FGD);  // Foreground mask
    fg_mask.convertTo(fg_mask, CV_64F); // Convert mask to CV_64F for further computation

    return fg_mask;
}

// Function to compute foreground and background probabilities
tuple<Mat, double, double, double> compute_foreground_background_probability(const Mat& image_rgb, const Mat& edges_refined) {
    Mat fg_prob = grabcut_foreground(image_rgb); // Get foreground probability

    // Normalize fg_prob to range [0, 1]
    fg_prob.convertTo(fg_prob, CV_64F, 1.0 / 255.0);  // Normalize to [0, 1]

    // Ensure both fg_prob and edges_refined are of the same type (CV_64F for floating point operations)
    if (fg_prob.type() != CV_64F) {
        fg_prob.convertTo(fg_prob, CV_64F);
    }
    Mat edges_refined_64F;
    edges_refined.convertTo(edges_refined_64F, CV_64F);
    
    // Normalize edges_refined to [0, 1]
    edges_refined_64F /= 255.0;

    // Calculate the foreground score (mean of foreground probability)
    double foreground_score = mean(fg_prob)[0];
    double background_score = 1.0 - foreground_score;  // Background is the complement

    // Calculate the edge-weighted foreground score (based on refined edges)
    double edge_weighted_fg = 0.0;
    if (sum(edges_refined_64F)[0] > 0) {  // Check if edges exist
        edge_weighted_fg = sum(fg_prob.mul(edges_refined_64F))[0] / sum(edges_refined_64F)[0];
    }

    return make_tuple(fg_prob, foreground_score, background_score, edge_weighted_fg);
}

int main(int ac, char** av) {
    // Check if the input image is provided (if input size is not 2, return -1 and show usage message)
    if (ac != 2) {
        cout << "Usage: ./grabcut_edge_detection <image_path>" << endl;
        return -1;
    }

    // Load the image (ensure the path is correct)
    std::cout << "Loading image: " << av[1] << std::endl;

    Mat image = imread(av[1], IMREAD_COLOR);  // Load your image here
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Convert image to grayscale for edge detection
    std::cout << "Processing image..." << std::endl;
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    // Apply multi-scale Canny for edge detection
    std::cout << "Applying multi-scale Canny edge detection..." << std::endl;
    Mat edges_refined = multi_scale_canny(gray_image);

    // Compute foreground and background probabilities
    std::cout << "Computing foreground and background probabilities..." << std::endl;
    Mat fg_prob;
    double foreground_score, background_score, edge_weighted_fg;
    tie(fg_prob, foreground_score, background_score, edge_weighted_fg) = compute_foreground_background_probability(image, edges_refined);

    // Print out the probabilities (scores)
    cout << "Foreground Probability Score: " << foreground_score << endl;
    cout << "Background Probability Score: " << background_score << endl;
    cout << "Edge-Weighted Foreground Score: " << edge_weighted_fg << endl;

    imshow("Foreground Probability", fg_prob);
    imshow("Refined Edges", edges_refined);
    waitKey(0);  // Wait for a key press to close the window

    return 0;
}
