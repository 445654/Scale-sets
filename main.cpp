#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include <unistd.h>

using namespace std;
using namespace cv;

using Pixel = pair<int, int>;

struct Region
{
    int id{0}; // TODO remove ?
    set<Pixel> pixels;
    set<int> adjacentRegions; // TODO remove ?
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

/**
 * Return the von neumann neighborhood of the pixel passed in parameters.
 *
 * @param pixel
 * @return
 */
vector<Pixel> getPixelNeighbors(Pixel pixel)
{
    vector<Pixel> neighbors;
    neighbors.push_back(Pixel(pixel.first - 1, pixel.second));
    neighbors.push_back(Pixel(pixel.first, pixel.second + 1));
    neighbors.push_back(Pixel(pixel.first + 1, pixel.second));
    neighbors.push_back(Pixel(pixel.first, pixel.second - 1));
    return neighbors;
}

/**
 * Return the ids of the neighbor regions of a region at its initial state (a pixel).
 *
 * @param initPixel region at its initial state
 * @param width image width
 * @param height image height
 * @return ids of the neighbor regions
 */
set<int> getInitRegionNeighbors(Pixel initPixel, int width, int height)
{
    set<int> neighbors;

    if (initPixel.first > 0)
        neighbors.emplace(initPixel.first * width + initPixel.second - width);
    if (initPixel.second < width - 1)
        neighbors.emplace(initPixel.first * width + initPixel.second + 1);
    if (initPixel.first < height - 1)
        neighbors.emplace(initPixel.first * width + initPixel.second + width);
    if (initPixel.second > 0)
        neighbors.emplace(initPixel.first * width + initPixel.second - 1);

    return neighbors;
}

/**
 * Return the perimeter length of the two regions if they were to be merged.
 * It's the sum of their original perimeters minus the length of their intersection.
 * Complexity of min(n_r1 log n_r2 ; n_r2 log n_r1).
 *
 * @param r1 region to merge
 * @param r2 region to merge
 * @return perimeter of the merged region
 */
int mergedPerimeter(Region r1, Region r2)
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
        vector<Pixel> neighbors = getPixelNeighbors(pixel);
        for (auto pixel : neighbors) // 4n
            if (bigRegion.pixels.find(pixel) != bigRegion.pixels.end()) // log(n)
                length++;
    }
    return length;
}

double energy(Mat image, Region r, double lambda)
{
    return variance(image, r) + lambda * r.perimeter;
}

Region simulatedMerge(Region* r1, Region* r2, vector<vector<int>> regionIds)
{
    Region outR1;

    // TODO copy pixels and adjacent regions to avoid side effect
    // copy r1
    outR1.id = r1->id;
    for (auto it = r1->pixels.begin(); it != r1->pixels.end(); ++it)
        outR1.pixels.emplace(*it);
    for (auto it = r1->adjacentRegions.begin(); it != r1->adjacentRegions.end(); ++it)
        outR1.adjacentRegions.emplace(*it);
    outR1.perimeter = r1->perimeter;

    // update r1
    outR1.pixels.merge(r2->pixels); // merge pixels
    outR1.perimeter = outR1.perimeter + r2->perimeter - mergedPerimeter(outR1, *r2); // update perimeter
    outR1.adjacentRegions.merge(r2->adjacentRegions); // add new adjacent regions

    return outR1;
}

/**
 * Merges two regions by adding the pixels of r2 into r1 without duplicating them.
 * r1 pixels are modified, r2's are not modified.
 * Returns the new region ids matrix.
 *
 * @param r1 will have r2 merged into
 * @param r2 will be merged into r1
 * @return new region ids matrix
 */
Region merge(Region* r1, Region* r2, vector<vector<int>> regionIds)
{
    Region outR1 = simulatedMerge(r1, r2, regionIds);

    // switch region ids to r1's for pixels which were in r2
    for (int i = 0; i < regionIds.size(); ++i)
        for (int j = 0; j < regionIds[i].size(); ++j)
            if (regionIds[i][j] == r2->id)
                regionIds[i][j] = r1->id;

    return outR1;
}

double optimalLambda(Region r1, Region r2, Mat image, vector<vector<int>> regionIds)
{
    double varA, varB, varAUB, perA, perB, perAUB, lambda;
    varA = variance(image, r1);
    varB = variance(image, r2);

    perA = r1.perimeter;
    perB = r2.perimeter;

    Region mergedRegion = simulatedMerge(&r1, &r2, regionIds);

    varAUB = variance(image, mergedRegion);
    perAUB = mergedRegion.perimeter;

    lambda = (varA + varB - varAUB) / (perA + perB - perAUB);

    return lambda;
}

void scaleSets(Mat input)
{
    vector<vector<int>> regionIds;
    for (int i = 0; i < input.rows; ++i)
    {
        regionIds.push_back(vector<int>());
        for (int j = 0; j < input.cols; ++j)
            regionIds[i].push_back(i * input.cols + j);
    }
    // create region for each pixel
    vector<Region> regions; // array of regions
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
        {
            regions.push_back(Region{
                    i * input.cols + j,
                    set<Pixel>{Pixel(i, j)},
                    getInitRegionNeighbors(Pixel(i, j), input.cols, input.rows),
                    4
            });
        }

    int nbRegions = input.rows * input.cols;
    int nbCount = 20000;
    int count = nbCount;
    while (nbRegions != 1 && count != 0)
    {
        Region r1, r2;
        double lambdaMin = 100000000.;
        // TODO : handle 'active' regions and optimise code (obj don't calculate the same distance twice)
        vector<int> doneRegion;
        // find min lambda and the merge associated with it
        for (int i = 0; i < regionIds.size(); ++i) {
            for (int j = 0; j < regionIds[0].size(); ++j) {
                if (find(doneRegion.begin(), doneRegion.end(), regionIds[i][j]) != doneRegion.end()) {
                    for (int neighborIdx: regions[regionIds[i][j]].adjacentRegions) {
                        if (find(doneRegion.begin(), doneRegion.end(), neighborIdx) != doneRegion.end()) {
                            Region neighbor = regions[neighborIdx];
                            // TODO: copy regions
                            double lambda = optimalLambda(regions[i], neighbor, input, regionIds);
                            if (lambda < lambdaMin) {
                                lambdaMin = lambda;
                                r1 = regions[i];
                                r2 = neighbor;
                            }
                        }
                    }
                    doneRegion.push_back(i * regionIds.size() + j);
                }
            }
        }
        r1 = merge(&r1,&r2, regionIds);
        if (count % (nbCount/100) == 100)
            cout << "end loop ! " << count << endl;
        count--;
        nbRegions--;
    }
    Mat output;
    input.copyTo(output);

    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            if (regions[regionIds.at(i).at(j)].pixels.size() != 1)
            {
                cout << regions[regionIds.at(i).at(j)].pixels.size() << endl;
                output.at<Vec3b>(i, j) = Vec3b(255,0,0); // bgr space
            }


    // show output image
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", output);
    //  find regions with min energy
    //  merge the two regions
}

int main()
{
    Mat input = imread("../images/pobo_small.png", IMREAD_COLOR);

    // show original image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input);

    scaleSets(input);

    Mat output;
    input.copyTo(output);

    // coloring merged regions in output
    /*for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
            if (regions[regionIds.at(i).at(j)].pixels.size() != 1)
            {
                //cout << regions[regionIds.at(i).at(j)].pixels.size() << endl;
                output.at<Vec3b>(i, j) = Vec3b(255,0,0); // bgr space
            }
    */

    // show output image
    //namedWindow("Output", WINDOW_AUTOSIZE);
    //imshow("Output", output);

    waitKey(0);
    return 0;
}
