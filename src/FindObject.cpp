#include "Common.h"
#include "FindObject.h"

//#define DEBUG_FIND_OBJECT

void findCurves(const std::vector<int>& histogram, std::vector<std::tuple<int, int, int>>& curves)
{
    bool inCurve = false;
    int start;
    int size;
    for (int i = 0; i < histogram.size(); i++)
    {
        // Attempt to find the start of a curve
        if (!inCurve)
        {
            if (histogram[i] > 0)
            {
                inCurve = true;
                start = i;
                size = histogram[i];
            }
        }
        else // Attempt to find the end of the curve
        {
            size += histogram[i];
            if (i == (histogram.size() - 1) || histogram[i + 1] == 0) // short circuiting will prevent array out of bounds
            {
                inCurve = false;
                curves.push_back(std::make_tuple(start, i, size));
            }
        }
    }
}

bool findObject(cv::Mat image, float depthScale, float& objectDistance)
{
    // Set up a histogram
    std::vector<int> counts;
    const int CLASS_COUNT = 20;
    counts.resize(CLASS_COUNT);

    // Map pixel to class number
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            uint16_t rawDepth = image.at<uint16_t>(r, c);

            // Ignore indeterminate values
            if (rawDepth == 0)
                continue;

            // Convert raw depth into 0-1 range
            float rawDepthScaled = (float)rawDepth / (float)0xffff;
            counts[(int)(rawDepthScaled * CLASS_COUNT)]++;
        }
    }

#ifdef DEBUG_FIND_OBJECT
    // Display histogram for debugging
    for (int c : counts)
        cout << c << ", " << endl;
    cout << endl;
#endif

    // Extract curves in form min, max (inclusive), area
    std::vector<std::tuple<int, int, int>> curves;
    findCurves(counts, curves);

    // If there are no curves, then there are no depth values
    if (curves.empty())
        return false;

    // Find the largest curve by area
    auto largestCurve = curves.begin();
    for (auto it = curves.begin(); it != curves.end(); it++)
    {
        if (std::get<2>(*it) > std::get<2>(*largestCurve))
            largestCurve = it;
    }

    // Find average of all points between the start and end of the first curve
    float sum = 0.0f;
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            uint16_t rawDepth = image.at<uint16_t>(r, c);

            // Ignore indeterminate values
            if (rawDepth == 0)
                continue;

            // Convert raw depth into class
            float rawDepthScaled = (float)rawDepth / (float)0xffff;
            int depthClass = (int)(rawDepthScaled * CLASS_COUNT);

            // If class is within range defined above, consider it
            if (depthClass >= std::get<0>(*largestCurve) && depthClass <= std::get<1>(*largestCurve))
                sum += rawDepth * depthScale;
        }
    }

    // Return average
    objectDistance = sum / std::get<2>(*largestCurve);
    return true;
}