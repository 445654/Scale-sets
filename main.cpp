#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

struct Region
{
    int id;
    vector<pair<int, int>> pixels;
    vector<int> adjacentRegions;

};

vector<Region> REGIONS;

double mean(const Mat &image, const Region &r)
{
    double sum;

    for (pair<int, int> pixel : r.pixels)
        sum += image.at<float>(pixel.first, pixel.second);

    return sum / r.pixels.size();
}

/**
 * Returns the variance for the pixels of the region passed in parameters.
 * Computes the variance for each channels and returns the means of these variances.
 *
 * @param image rgb image
 * @param r pixels region
 * @return variance of that region
 */
double variance(Mat image, Region r)
{
    double variances[3];

    // color channels split
    Mat channels[3];
    split(image, channels);

    // variance for each channels
    for (int i = 0; i < 3; ++i)
    {
        double sum = 0.;
        double meanChannel = mean(channels[i], r);
        for (pair<int, int> pixel : r.pixels)
            sum += pow(channels[i].at<float>(pixel.first, pixel.second) - meanChannel, 2);
        variances[i] = sum / r.pixels.size();
    }

    // mean of channels variances
    double varianceSum = 0;
    for (double variance : variances)
        varianceSum += variance;

    return varianceSum / 3;
}

vector<pair<int,int>> findBorder(Region r)
{
    vector<pair<int,int>> border;
    for (pair<int,int> pixel : r.pixels)
    {
        pair<int,int> neighbors[4];
        neighbors[0] = pair<int,int>(pixel.first - 1, pixel.second);
        neighbors[1] = pair<int,int>(pixel.first, pixel.second + 1);
        neighbors[2] = pair<int,int>(pixel.first + 1, pixel.second);
        neighbors[3] = pair<int,int>(pixel.first, pixel.second - 1);

        for (pair<int,int> neighbor : neighbors)
            if (find(r.pixels.begin(), r.pixels.end(), neighbor) != r.pixels.end())
            {
                border.push_back(pixel);
                break;
            }
    }
    return border;
}

double perimeter(Mat image, Region r)
{
    double perimeter = 0.;

    // border detection
    vector<pair<int,int>> border = findBorder(r);

    return perimeter;
}

double energy(Mat image, Region r, double lambda)
{
    return variance(image, r) + lambda * perimeter(image, r);
}

Region merge(Region r1, Region r2)
{
    Region result;



    return result;
}

int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
