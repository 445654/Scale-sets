#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <map>

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
    int size{0};
    set<int> adjacentRegions;
    int perimeter{0};
    double s{0.};
    double t{0.};
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
double variance(Region &region)
{
    double n = region.size;
    return (region.t / n)  - pow(region.s / n, 2);
}

double variance(const Mat &im, Region &region)
{
    double variances;

    // variance for each channels
    double sum = 0.;
    double meanChannel = mean(im, region);
    for (Pixel pixel: region.pixels) // n_r
        sum += pow(((double) im.at<float>(pixel.first, pixel.second)) - meanChannel, 2);
    variances = sum / (double) region.pixels.size();

    return variances ;
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
set<int> getInitRegionNeighbors(const Pixel& initPixel, const int& width, const int& height)
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

    mergedRegion.perimeter = region2.perimeter + region1.perimeter + mergedPerimeter(region1, region2);

    int n = region1.size;
    int n_p = region2.size;

    mergedRegion.size = region1.size + region2.size;

    mergedRegion.t = (region1.t + region2.t);
    //mergedRegion.t = (n * region1.t + n_p * region2.t) / mergedRegion.size;
    mergedRegion.s = (region1.s + region2.s); // / mergedRegion.size
    //mergedRegion.s = (n * region1.s + n_p * region2.s) / mergedRegion.size; // / mergedRegion.size


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
double optimalLambda(Region &region1, Region &region2, const Mat& im)
{
    double tmp1 = variance(im, region1), tmp2 = variance(region1);
    assert(tmp1 == tmp2);
    double tmp3 = variance(im, region2), tmp4 = variance(region2);
    assert(tmp3 == tmp4);
    double variance1 = variance(im, region1);
    double variance2 = variance(im, region2);

    double perimeter1 = region1.perimeter;
    double perimeter2 = region2.perimeter;

    // merge region2 into region1
    Region mergedRegion = region1; // n_r1
    set<Pixel> region2Pixels = region2.pixels; // n_r2
    mergedRegion.pixels.merge(region2Pixels); // n_r1 * log (n_r1 + n_r2)

    // merged region lambda
    double n = region1.size;
    double n_p = region2.size;
    mergedRegion.t = (region1.t + region2.t);
    mergedRegion.s = (region1.s + region2.s);
    mergedRegion.size = n + n_p;
    double variance1U2 = variance(mergedRegion) ;//= (t / (n + n_p) - pow(s / (n_p + n), 2)); // / (n + n_p)
    double perimeter1U2 = perimeter1 + perimeter2 - mergedPerimeter(region1, region2); // min(n_r1 log n_r2 ; n_r2 log n_r1)
    double lambda = (variance1 + variance2 - variance1U2) / (perimeter1U2 - perimeter1 - perimeter2);

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
            output.at<float>(i, j) = encounteredRegionsColors[regionIndex][0];
        }

    // show merged regions image
    Mat resized_output;
    resize(output, resized_output, Size(), 100.0, 100.0, CV_INTER_NN);
    namedWindow("Merged regions" ,  WINDOW_AUTOSIZE);
    imshow("Merged regions", resized_output);
}

void scaleSets(const Mat &input)
{
    vector<Region> regions; // array of regions
    vector<int> activeRegions; // array of regions present in image
    vector<vector<int>> regionIds; // TODO use a fixed size array instead of a vector

    // split input image into 3 channels
    /*Mat tmp[3];
    split(input, tmp);
    vector<Mat> channels;
    channels.push_back(tmp[0]);
    channels.push_back(tmp[1]);
    channels.push_back(tmp[2]);//*/
    int imageSize = input.rows * input.cols;
    std::map<std::pair<int,int>, double> lambdaMatrix;

    // create a region for each pixel
    for (int i = 0; i < input.rows; ++i)
    {
        regionIds.emplace_back();
        for (int j = 0; j < input.cols; ++j)
        {
            int id = i * input.cols + j;
            Region r = Region{
                    id,
                    set<Pixel>{Pixel(i, j)},
                    1,
                    getInitRegionNeighbors(Pixel(i, j), input.cols, input.rows),
                    4
            };

            //for (int k = 0; k < 3; ++k) {
                r.t += pow(input.at<float>(i, j), 2);
                r.s += input.at<float>(i, j);
            //}
            //r.t /= 3;
            //r.s /= 3;
            regions.emplace_back(r);
            regionIds[i].push_back(id);
            activeRegions.push_back(id);
        }
    }

    for (int regionId: activeRegions)
    {
        for (int neighborIdx: regions[regionId].adjacentRegions)
        {
            double lambda = optimalLambda(regions[regionId], regions[neighborIdx], input);
            lambdaMatrix.emplace(pair<int,int>(regionId,neighborIdx), lambda);
            lambdaMatrix.emplace(pair<int,int>(neighborIdx, regionId), lambda);
        }
    }


    int nbCount = 12;
    int count = nbCount;
    time_t timeAtLoopStart, timeAfterActiveRegionsPassed, timeAfterRemove, timeAfterEverything, totalItTime = 0, totalTimeLoop = 0, totalTimeRemove = 0, totalTimeUpdate = 0;
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
                //int tmp = neighborIdx % input.rows;
                //int tmp2 = (neighborIdx-tmp) / input.rows;
                //int trueNI = regionIds.at(tmp2).at(tmp);
                // neighbor is not in doneRegions
                // TODO : update indexs (problem : if neighbor isn't active anymore)
                // x
                // we use the fact that the activeRegions is sorted
                if (neighborIdx > regionId)
                {
                    double lambda = lambdaMatrix.at(pair<int,int>(regionId,neighborIdx));
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

        /*cout << "=========== " << "merge" << " ===========" << endl;
        cout << "t: " << r1.t << " s: " << r1.s << endl;
        for (auto pix : r1.pixels)
            cout << pix.first << " " << pix.second << " : ";
        cout << endl;
        cout << "t: " << r2.t << " s: " << r2.s << endl;
        for (auto pix : r2.pixels)
            cout << pix.first << " " << pix.second << " : ";
        cout << endl;
        cout << "lmin: " << lambdaMin << endl;*/

        int newRegionId = r1.id;
        // r2 is merged into r1, only r1 remains
        regions[newRegionId] = merge(r1, r2, regionIds);
        for (int neighborId : r2.adjacentRegions)
        {
            lambdaMatrix.erase(pair<int,int>(r2.id, neighborId));
            lambdaMatrix.erase(pair<int,int>(neighborId, r2.id));
        }

        timeAfterRemove = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        // update neighbors of merged region :
        // delete r2 and add r1 in the neighbors list of merged region neighbors
        for (int neighborId: regions[newRegionId].adjacentRegions)
        {
            double lambda = optimalLambda(regions[newRegionId], regions[neighborId], input);
            lambdaMatrix.emplace(pair<int,int>(newRegionId, neighborId), lambda);
            lambdaMatrix.emplace(pair<int,int>(neighborId, newRegionId), lambda);
            // check if neighborId is also a neighbor of r2
            // TODO : avoid this update
            if (regions[neighborId].adjacentRegions.count(r2.id) != 0)
            {
                regions[neighborId].adjacentRegions.erase(r2.id);
                if (regions[neighborId].adjacentRegions.count(newRegionId) == 0)
                    regions[neighborId].adjacentRegions.insert(newRegionId);
            }
        }
        totalNeighbors += regions[newRegionId].adjacentRegions.size();

        activeRegions.erase(find(activeRegions.begin(), activeRegions.end(), r2.id));

        timeAfterEverything = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        totalItTime += (timeAfterEverything - timeAtLoopStart);
        totalTimeLoop += (timeAfterActiveRegionsPassed - timeAtLoopStart);
        totalTimeRemove += (timeAfterRemove - timeAfterActiveRegionsPassed);
        totalTimeUpdate += (timeAfterEverything - timeAfterActiveRegionsPassed);
        //cout << "duration of loop : " << (timeAfterActiveRegionsPassed - timeAtLoopStart) << " duration of the rest : "
        //     << (timeAfterEverything - timeAfterActiveRegionsPassed) << endl;
        if (count % 100 == 0) {
            cout << count
                 << " total : " << totalItTime
                 << " loop : " << totalTimeLoop
                 << " remove : " << totalTimeRemove
                 << " rest : " << totalTimeUpdate
                 << " neighbors : " << totalNeighbors
                 << " lambda size : " << lambdaMatrix.size() << endl;
            totalItTime = 0;
            totalTimeLoop = 0;
            totalTimeUpdate = 0;
            totalNeighbors = 0;
            totalTimeRemove = 0;
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
    cv::cvtColor( input, input, COLOR_BGR2GRAY );
    input.convertTo(input, CV_32FC1, 1.0 / 255.0); // convert image to float type

    scaleSets(input);

    namedWindow("Original", WINDOW_AUTOSIZE);
    resize(input, input, Size(), 100.0, 100.0, CV_INTER_NN);
    imshow("Original", input);

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
