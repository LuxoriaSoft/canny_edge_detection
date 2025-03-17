#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>

class GrabCut {
    public:
        static std::tuple<cv::Mat, double, double, double> ComputeForegroundProbability(const cv::Mat& image);

    private:
        static cv::Mat MultiScaleCanny(const cv::Mat& image);
        static cv::Mat SimpleGrabCutForeground(const cv::Mat& image);
};
