#pragma once

#include <tuple>

void findCurves(const std::vector<int>& histogram, std::vector<std::tuple<int, int, int>>& curves);
bool findObject(cv::Mat image, float depthScale, float& objectDistance);
