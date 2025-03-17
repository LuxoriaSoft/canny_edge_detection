#pragma once

#include <opencv2/opencv.hpp>
#include <msclr/marshal_cppstd.h>
using namespace System;
using namespace System::Drawing;
using namespace System::Runtime::InteropServices;

public ref class GrabCutWrapper {
public:
    static array<double, 2>^ ComputeForegroundProbability(String^ imagePath) {
        std::string imgPath = msclr::interop::marshal_as<std::string>(imagePath);
        cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw gcnew ArgumentException("Could not open or find the image!");
        }
        
        if (image.cols < 100 || image.rows < 100) {
            cv::resize(image, image, cv::Size(100, 100));
        }
        
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        cv::Mat edges_refined = MultiScaleCanny(gray_image);
        
        cv::Mat fg_prob;
        double foreground_score, background_score, edge_weighted_fg;
        std::tie(fg_prob, foreground_score, background_score, edge_weighted_fg) = ComputeForegroundBackgroundProbability(image, edges_refined);
        
        array<double, 2>^ result = gcnew array<double, 2>(3, 1);
        result[0, 0] = foreground_score;
        result[1, 0] = background_score;
        result[2, 0] = edge_weighted_fg;
        return result;
    }

private:
    static cv::Mat MultiScaleCanny(const cv::Mat& image) {
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

    static std::tuple<cv::Mat, double, double, double> ComputeForegroundBackgroundProbability(
        const cv::Mat& image_rgb, const cv::Mat& edges_refined) {
        cv::Mat fg_prob = SimpleGrabCutForeground(image_rgb);
        fg_prob = (fg_prob > 0);
        fg_prob.convertTo(fg_prob, CV_64F);
        
        double minVal, max_prob;
        cv::minMaxLoc(fg_prob, &minVal, &max_prob);
        
        if (max_prob > 0) {
            fg_prob /= max_prob;
        }
        
        double foreground_score = cv::mean(fg_prob)[0];
        double background_score = 1.0 - foreground_score;
        
        cv::Mat edges_refined_64F;
        edges_refined.convertTo(edges_refined_64F, CV_64F);
        edges_refined_64F /= 255.0;
        
        double edge_weighted_fg = cv::sum(fg_prob.mul(edges_refined_64F))[0] / 
                                  std::max(1.0, cv::sum(edges_refined_64F)[0]);

        return std::make_tuple(fg_prob, foreground_score, background_score, edge_weighted_fg);
    }

    static cv::Mat SimpleGrabCutForeground(const cv::Mat& image) {
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        
        cv::Mat mask;
        cv::threshold(gray_image, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        mask.convertTo(mask, CV_64F);
        
        return mask;
    }
};
