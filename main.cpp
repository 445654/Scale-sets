#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <set>

using namespace std;
using namespace cv;

using Pixel = pair<int, int>;

struct Region
{
    int id{0};
    set<Pixel> pixels;
    set<int> adjacentRegions;
    int perimeter{0};
};

double mean(const Mat &image, const Region &r)
{
    double sum;

    for (Pixel pixel : r.pixels)
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
        for (Pixel pixel : r.pixels)
            sum += pow(channels[i].at<float>(pixel.first, pixel.second) - meanChannel, 2);
        variances[i] = sum / r.pixels.size();
    }

    // mean of channels variances
    double varianceSum = 0;
    for (double variance : variances)
        varianceSum += variance;

    return varianceSum / 3;
}

Pixel* getNeighbors(Pixel pixel)
{
    Pixel neighbors[4];
    neighbors[0] = Pixel(pixel.first - 1, pixel.second);
    neighbors[1] = Pixel(pixel.first, pixel.second + 1);
    neighbors[2] = Pixel(pixel.first + 1, pixel.second);
    neighbors[3] = Pixel(pixel.first, pixel.second - 1);
    return neighbors;
}

/**
 * Return the perimeter of the two regions if they were to be merged.
 * It's the sum of their original perimeters minus the length of their intersection.
 * Complexity of min(n_r1 log n_r2 ; n_r2 log n_r1).
 *
 * @param r1 region to merge
 * @param r2 region to merge
 * @return perimeter of the merged region
 */
int intersectionLength(Region r1, Region r2)
{
    // checking which region is smaller
    Region smallRegion;
    Region bigRegion;
    if (r1.pixels.size() < r2.pixels.size())
    {
        smallRegion = r1;
        bigRegion = r2;
    }
    else
    {
        smallRegion = r2;
        bigRegion = r1;
    }

    // looping through smallest region to check
    // intersection pixels with bigger region
    int length = 0;
    for (Pixel pixel : smallRegion.pixels) // n
    {
        Pixel* neighbors = getNeighbors(pixel);
        for (int i = 0; i < 4; ++i) // 4n
            if (bigRegion.pixels.find(neighbors[i]) != bigRegion.pixels.end()) // log(n)
                length++;
    }
    return length;
}

double perimeter(Mat image, Region r)
{
    double perimeter = 0.;

    // border detection
    //vector<Pixel> border = findBorder(r);

    return perimeter;
}

double energy(Mat image, Region r, double lambda)
{
    return variance(image, r) + lambda * perimeter(image, r);
}

/**
 * Merges two regions by adding the pixels of r2 into r1 without duplicating them.
 *
 * @param r1 will have r2 merged into
 * @param r2 will be merged into r1
 */
void merge(Region r1, Region r2)
{
    Region result;

    r1.pixels.merge(r2.pixels);
    r1.perimeter = r1.perimeter + r2.perimeter - intersectionLength(r1, r2);

    r1.adjacentRegions.merge(r2.adjacentRegions);
}

int main()
{
    cout << "Hello, World!" << endl;

    Region r1;
    Region r2;

    r1.adjacentRegions.emplace(4);
    r2.adjacentRegions.emplace(4);
    r2.adjacentRegions.emplace(4);
    r2.adjacentRegions.emplace(4);
    r2.adjacentRegions.emplace(4);
    r2.adjacentRegions.emplace(5);

    r1.adjacentRegions.merge(r2.adjacentRegions);

    cout << "r1 :" << endl;
    for (auto a : r1.adjacentRegions)
        cout << a << " ";
    cout << endl;
    cout << "r2 :" << endl;
    for (auto a : r2.adjacentRegions)
        cout << a;
    cout << endl;

    return 0;
}
