#pragma once

#include <opencv2/opencv.hpp>
#include <msclr/marshal_cppstd.h>

using namespace System;

public ref class GrabCutWrapper {
public:
    static array<double, 2>^ ComputeForegroundProbability(String^ imagePath);

private:
    static cv::Mat MultiScaleCanny(const cv::Mat& image);
    static std::tuple<cv::Mat, double, double, double> ComputeForegroundBackgroundProbability(
        const cv::Mat& image_rgb, const cv::Mat& edges_refined);
    static cv::Mat SimpleGrabCutForeground(const cv::Mat& image);
};
