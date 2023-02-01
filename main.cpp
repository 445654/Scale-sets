#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include <chrono>

using namespace std;
using namespace cv;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

using Pixel = pair<int, int>;

struct Region
{
    int id{-1};
    set<Pixel> pixels;
    set<int> adjacentRegions;
    int perimeter{0};
    double variance{};
};

double mean(const Mat &image, Region &region)
{
    double sum;
    for (Pixel pixel: region.pixels)
        sum += image.at<float>(pixel.first, pixel.second);
    return sum / (double) region.pixels.size();
}

/**
 * Returns the variance for the pixels of the region passed in parameters.
 * Computes the variance for each channels and returns the means of these variances.
 *
 * @param image rgb image
 * @param region pixels region
 * @return variance of that region
 */
double variance(const vector<Mat> &channels, Region &region)
{
    double variances[3];

    // variance for each channels
    for (int i = 0; i < 3; ++i)
    {
        double sum = 0.;
        double meanChannel = mean(channels[i], region);
        for (Pixel pixel: region.pixels) // n_r
            sum += pow(((double) channels[i].at<float>(pixel.first, pixel.second)) - meanChannel, 2);
        variances[i] = sum / (double) region.pixels.size();
    }

    // mean of channels variances
    double varianceSum = 0;
    for (double variance: variances)
        varianceSum += variance;

    return varianceSum / 3;
}

/**
 * Return the von neumann neighborhood of the pixel passed in parameters.
 *
 * @param pixel
 * @return von neumann neighborhood
 */
vector<Pixel> getPixelNeighbors(Pixel pixel)
{
    vector<Pixel> neighbors;

    neighbors.emplace_back(pixel.first - 1, pixel.second);
    neighbors.emplace_back(pixel.first, pixel.second + 1);
    neighbors.emplace_back(pixel.first + 1, pixel.second);
    neighbors.emplace_back(pixel.first, pixel.second - 1);

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
 * @param region1 region to merge
 * @param region2 region to merge
 * @return perimeter of the merged region
 */
int mergedPerimeter(const Region &region1, const Region &region2)
{
    // checking which region is smaller
    Region smallRegion;
    Region bigRegion;
    if (region1.pixels.size() < region2.pixels.size())
    {
        smallRegion = region1;
        bigRegion = region2;
    }
    else
    {
        smallRegion = region2;
        bigRegion = region1;
    }

    // looping through smallest region to check
    // intersection pixels with bigger region
    int length = 0;
    for (Pixel pixel: smallRegion.pixels) // n
    {
        vector<Pixel> neighbors = getPixelNeighbors(pixel);
        for (auto neighbor: neighbors) // 4n
            if (bigRegion.pixels.find(neighbor) != bigRegion.pixels.end()) // log(n)
                length++;
    }
    return length;
}

/**
 * Merges two regions by adding the pixels of region2 into region1 without duplicating them.
 * region1 pixels are modified, region2's are not modified.
 * Returns the new region ids matrix.
 * Updates regionIds matrix.
 *
 * @param region1 will have region2 merged into
 * @param region2 will be merged into region1
 * @param regionIds region ids correspondence matrix, is modified after execution
 * @return new region ids matrix
 */
Region merge(const Region &region1, Region region2, vector<vector<int>> &regionIds)
{
    // merge region2 into region1
    Region mergedRegion = region1;
    mergedRegion.pixels.merge(region2.pixels); // merge pixels
    mergedRegion.adjacentRegions.merge(region2.adjacentRegions); // add new adjacent regions
    mergedRegion.adjacentRegions.erase(
            find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region2.id));
    mergedRegion.adjacentRegions.erase(
            find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region1.id));

    mergedRegion.perimeter = mergedPerimeter(region1, region2);

    // switch region ids to region1's for pixels which were in region2
    for (auto &regionId: regionIds)
        for (int &id: regionId)
            if (id == region2.id)
                id = region1.id;

    return mergedRegion;
}

/**
 * Returns the optimal lambda value for the two regions.
 * The optimal lambda value is the one that minimizes the energy function.
 *
 * @param region1
 * @param region2
 * @param channels
 * @return optimal lambda value
 */
double optimalLambda(Region &region1, Region &region2, const vector<Mat> &channels)
{
    double variance1 = region1.variance;
    double variance2 = region2.variance;

    double perimeter1 = region1.perimeter;
    double perimeter2 = region2.perimeter;

    // merge region2 into region1
    Region mergedRegion = region1; // n_r1
    set<Pixel> region2Pixels = region2.pixels; // n_r2
    mergedRegion.pixels.merge(region2Pixels); // n_r1 * log (n_r1 + n_r2)

    // merged region lambda
    double variance1U2 = variance(channels, mergedRegion); // n_r1 + n_r2
    double perimeter1U2 = mergedPerimeter(region1, region2); // min(n_r1 log n_r2 ; n_r2 log n_r1)
    double lambda = (variance1 + variance2 - variance1U2) / (perimeter1 + perimeter2 - perimeter1U2);

    return lambda;
}

/**
 * Displays the regions of the image with random colors.
 *
 * @param input input image
 * @param regions regions of the image
 * @param regionIds correspondence table between pixels and regions
 */
void displayRegions(const Mat &input, vector<Region> regions, vector<vector<int>> regionIds)
{
    Mat output;
    input.copyTo(output);

    vector<int> encounteredRegions;
    vector<Vec3f> encounteredRegionsColors;

    // coloring regions in output with a new random color for each region
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
        {
            int regionId = regions[regionIds[i][j]].id;
            auto it = find(encounteredRegions.begin(), encounteredRegions.end(), regionId);

            // create a new color if region has not been yet encountered
            if (it == encounteredRegions.end())
            {
                encounteredRegions.push_back(regionId);
                encounteredRegionsColors.emplace_back(
                        (rand() % 255) / 255.f,
                        (rand() % 255) / 255.f,
                        (rand() % 255) / 255.f
                );
            }

            // paint output with corresponding color
            it = find(encounteredRegions.begin(), encounteredRegions.end(), regionId);
            int regionIndex = (int) distance(encounteredRegions.begin(), it);
            output.at<Vec3f>(i, j) = encounteredRegionsColors[regionIndex];
        }

    // show merged regions image
    namedWindow("Merged regions", WINDOW_AUTOSIZE);
    imshow("Merged regions", output);
}

void scaleSets(const Mat &input)
{
    vector<Region> regions; // array of regions
    vector<int> activeRegions; // array of regions present in image
    vector<vector<int>> regionIds; // TODO use a fixed size array instead of a vector

    // split input image into 3 channels
    Mat tmp[3];
    split(input, tmp);
    vector<Mat> channels;
    channels.push_back(tmp[0]);
    channels.push_back(tmp[1]);
    channels.push_back(tmp[2]);
    int imageSize = input.rows * input.cols;
    vector<vector<double>> lambdaMatrix;

    for (int i = 0; i < input.rows * input.cols; ++i) {
        lambdaMatrix.emplace_back(vector<double>());
        for (int j = 0; j < input.rows * input.cols; ++j) {
            lambdaMatrix[i].emplace_back(0);
        }
    }

    // create a region for each pixel
    for (int i = 0; i < input.rows; ++i)
    {
        regionIds.emplace_back();
        for (int j = 0; j < input.cols; ++j)
        {
            int id = i * input.cols + j;
            regions.emplace_back(Region{
                    id,
                    set<Pixel>{Pixel(i, j)},
                    getInitRegionNeighbors(Pixel(i, j), input.cols, input.rows),
                    4
            });
            regions[j].variance = variance(channels, regions[j]);
            regionIds[i].push_back(id);
            activeRegions.push_back(id);
        }
    }

    for (int regionId: activeRegions)
    {
        for (int neighborIdx: regions[regionId].adjacentRegions)
        {
            double lambda = optimalLambda(regions[regionId], regions[neighborIdx], channels);
            lambdaMatrix[regionId][neighborIdx] = lambda;
            lambdaMatrix[neighborIdx][regionId] = lambda;
        }
    }


    int nbCount = 1200;
    int count = nbCount;
    time_t timeAtLoopStart, timeAfterActiveRegionsPassed, timeAfterEverything, totalItTime = 0, totalTimeLoop = 0, totalTimeUpdate = 0;
    double totalNeighbors = 0;
    while (activeRegions.size() > 1 && count != 0)
    {
        timeAtLoopStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        Region r1, r2;
        double lambdaMin = 100000000.;
        // TODO : handle 'active' regions and optimise code (obj don't calculate the same distance twice)
        vector<int> doneRegions;

        // find minimal lambda and the merge region associated with it
        // loop through all present regions
        // N - x
        for (int regionId: activeRegions)
        {
            // ~4
            for (int neighborIdx: regions[regionId].adjacentRegions)
            {
                // neighbor is not in doneRegions
                // TODO : update indexs (problem : if neighbor isn't active anymore)
                // x
                // we use the fact that the activeRegions is sorted
                if (neighborIdx > regionId)
                {
                    double lambda = lambdaMatrix[regionId][neighborIdx];
                    if (lambda < lambdaMin)
                    {
                        lambdaMin = lambda;
                        r1 = regions[regionId];
                        r2 = regions[neighborIdx];
                    }
                }
            }
            doneRegions.push_back(regionId);
        }
        timeAfterActiveRegionsPassed = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        int newRegionId = r1.id;
        // r2 is merged into r1, only r1 remains
        regions[newRegionId] = merge(r1, r2, regionIds);

        regions[newRegionId].variance = variance(channels, regions[newRegionId]);

        // update neighbors of merged region :
        // delete r2 and add r1 in the neighbors list of merged region neighbors
        for (int neighborId: regions[newRegionId].adjacentRegions)
        {
            double lambda = optimalLambda(regions[newRegionId], regions[neighborId], channels);
            lambdaMatrix[newRegionId][neighborId] = lambda;
            lambdaMatrix[neighborId][newRegionId] = lambda;
            // check if neighborId is also a neighbor of r2
            if (regions[neighborId].adjacentRegions.count(r2.id) != 0)
            {
                regions[neighborId].adjacentRegions.erase(r2.id);
                regions[neighborId].adjacentRegions.insert(newRegionId);
            }
        }
        totalNeighbors += regions[newRegionId].adjacentRegions.size();

        activeRegions.erase(find(activeRegions.begin(), activeRegions.end(), r2.id));

        timeAfterEverything = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        totalItTime += (timeAfterEverything - timeAtLoopStart);
        totalTimeLoop += (timeAfterActiveRegionsPassed - timeAtLoopStart);
        totalTimeUpdate += (timeAfterEverything - timeAfterActiveRegionsPassed);
        //cout << "duration of loop : " << (timeAfterActiveRegionsPassed - timeAtLoopStart) << " duration of the rest : "
        //     << (timeAfterEverything - timeAfterActiveRegionsPassed) << endl;
        if (count % 100 == 0) {
            cout << count << " total : "
                 << totalItTime << " loop : " << totalTimeLoop
                 << " rest : " << totalTimeUpdate
                 << " neighbors : " << totalNeighbors << endl;
            totalItTime = 0;
            totalTimeLoop = 0;
            totalTimeUpdate = 0;
            totalNeighbors = 0;
        }
        count--;
    }

    displayRegions(input, regions, regionIds);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <path_to_image>" << endl;
        return -1;
    }

    Mat input = imread(argv[1], IMREAD_COLOR);
    input.convertTo(input, CV_32FC3, 1.0 / 255.0); // convert image to float type

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input);

    scaleSets(input);

    Mat output;
    input.copyTo(output);

    while (true)
    {
        int keycode = waitKey(50);
        int asciiCode = keycode & 0xff;
        if (asciiCode == 'q')
            break;
    }

    return 0;
}
