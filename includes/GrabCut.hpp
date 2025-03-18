#pragma once
#include <opencv2/opencv.hpp>

// Declare the export function for use in .NET
extern "C" __declspec(dllexport) void ComputeForegroundProbability(const char* imagePath, double* foregroundScore, double* backgroundScore);
