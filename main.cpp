#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <set>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <map>
#include <queue>
#include <argparse/argparse.hpp>

using namespace std;
using namespace cv;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

using Pixel = pair<int, int>;

const int COLOR_DIM = 3;

struct Elt
{
    int rank;
    int root;
};

int find_(map<int, Elt> &set, int value)
{
    if (set[value].root == value)
        return value;

    set[value].root = find_(set, set[value].root);
    return set[value].root;
}

void union_(map<int, Elt> &thingy, int v1, int v2)
{
    int tv1 = find_(thingy, v1);
    int tv2 = find_(thingy, v2);

    auto it1 = thingy.find(tv1);
    auto it2 = thingy.find(tv2);

    it2->second.root = it1->second.root;
}

struct Region
{
    int id{-1};
    set<Pixel> pixels;
    int size{0};
    set<int> adjacentRegions;
    int perimeter{0};
    double ePixels[COLOR_DIM]{0, 0, 0};
    double eSquares[COLOR_DIM]{0, 0, 0};

    bool operator<(const Region &other) const;
};

bool Region::operator<(const Region &other) const
{
    return this->id < other.id;
}

struct MergeCandidate
{
    int r1;
    int r2;
    double lambda;
    int perimeter1;
    int perimeter2;

    bool operator<(const MergeCandidate &other) const;
};

bool MergeCandidate::operator<(const MergeCandidate &other) const
{
    return this->lambda > other.lambda;
}

double mean(const Mat &image, const Region &region)
{
    double sum;
    for (Pixel pixel: region.pixels)
        sum += image.at<float>(pixel.first, pixel.second);
    return sum / (double) region.pixels.size();
}

void initEPixels(double *ePixels, const Vec3f &pixel)
{
    for (int channel = 0; channel < COLOR_DIM; channel++)
        ePixels[channel] = pixel[channel];
}

void initESquares(double *eSquares, const Vec3f &pixel)
{
    for (int channel = 0; channel < COLOR_DIM; channel++)
        eSquares[channel] = pixel[channel] * pixel[channel];
}

/**
 * Returns the variance for the pixels of the region passed in parameters.
 * Computes the variance for each channels and returns the means of these variances.
 *
 * @param image rgb image
 * @param region pixels region
 * @return variance of that region
 */
double variance(const Region &region)
{
    double n = region.size;
    double sum = 0;
    for (int channel = 0; channel < COLOR_DIM; channel++)
    {
        double channelVariance = 1 / n * (region.eSquares[channel] - pow(region.ePixels[channel], 2) / n);
        sum += abs(channelVariance);
    }
    return sum;
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
set<int> getInitRegionNeighbors(const Pixel &initPixel, const int &width, const int &height)
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
 * Return the length of the perimeter intersection of two regions.
 * It's the sum of their original perimeters minus the length of their intersection.
 * Complexity of min(n_r1 log n_r2 ; n_r2 log n_r1).
 *
 * @param region1 region to merge
 * @param region2 region to merge
 * @return perimeter of the merged region
 */
int intersectionPerimeter(const Region &region1, const Region &region2)
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

void *addArrays(double *output, const double *arr1, const double *arr2, int size)
{
    for (int i = 0; i < size; i++)
        output[i] = arr1[i] + arr2[i];
}

/**
 * Merges two regions by adding the pixels of region2 into region1 without duplicating them.
 * region1 pixels are modified, region2's are not modified.
 * Returns the new region ids matrix.
 * Updates regionIds matrix.
 *
 * @param region1 will have region2 merged into
 * @param region2 will be merged into region1
 * @return new region ids matrix
 */
Region merge(const Region &region1, Region &region2)
{
    // merge region2 into region1
    Region mergedRegion = region1;
    mergedRegion.pixels.merge(region2.pixels); // merge pixels
    mergedRegion.adjacentRegions.merge(region2.adjacentRegions); // add new adjacent regions

    // erasing neighbors pointing at the regions which has been merged
//    mergedRegion.adjacentRegions.erase(
//            find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region2.id));
//    mergedRegion.adjacentRegions.erase(
//            find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region1.id));

    mergedRegion.perimeter = region2.perimeter + region1.perimeter - intersectionPerimeter(region1, region2) * 2;
    mergedRegion.size = region1.size + region2.size;

    addArrays(mergedRegion.eSquares, region1.eSquares, region2.eSquares, COLOR_DIM);
    addArrays(mergedRegion.ePixels, region1.ePixels, region2.ePixels, COLOR_DIM);

    return mergedRegion;
}

/**
 * Returns the optimal lambda value for the two regions.
 * The optimal lambda value is the one that minimizes the energy function.
 *
 * @param region1
 * @param region2
 * @return optimal lambda value
 */
double optimalLambda(const Region &region1, const Region &region2)
{
    double variance1 = variance(region1);
    double variance2 = variance(region2);

    double perimeter1 = region1.perimeter;
    double perimeter2 = region2.perimeter;

    // simulate merging region2 into region1
    Region mergedRegion = region1;
    set<Pixel> region2Pixels = region2.pixels;
    mergedRegion.pixels.merge(region2Pixels);
    mergedRegion.size = region1.size + region2.size;
    addArrays(mergedRegion.eSquares, region1.eSquares, region2.eSquares, COLOR_DIM);
    addArrays(mergedRegion.ePixels, region1.ePixels, region2.ePixels, COLOR_DIM);

    // merged region lambda
    double variance1U2 = variance(mergedRegion);
    double perimeter1U2 = perimeter1 + perimeter2 - intersectionPerimeter(region1, region2) * 2;
    double lambda = (variance1U2 - variance1 - variance2) / (perimeter1 + perimeter2 - perimeter1U2);

    return lambda;
}

/**
 * Displays the regions of the image with random colors.
 *
 * @param input input image
 * @param regions regions of the image
 * @param regionIds correspondence table between pixels and regions
 */
void displayRegions(const Mat &input, map<int, Region> regions, map<int, Elt> &union_find_set)
{
    Mat output;
    input.copyTo(output);
    map<int, vector<float>> encounteredRegions;

    // coloring regions in output with a new random color for each region
    auto itPixel = union_find_set.begin();
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
        {
            int regionId = regions[find_(union_find_set, itPixel->first)].id;
            auto it = encounteredRegions.find(regionId);

            // create a new color if region has not been yet encountered
            if (it == encounteredRegions.end())
            {
                auto res = encounteredRegions.emplace(
                        regionId,
                        vector{rand() % 255 / 255.f, rand() % 255 / 255.f, rand() % 255 / 255.f}
                );
                if (!res.second)
                {
                    cout << "warning! failed to emplace" << endl;
                    exit(1);
                }
                it = res.first;
            }

            // paint output with corresponding color
            output.at<Vec3f>(i, j) = Vec3f{it->second.at(0), it->second.at(1), it->second.at(2)};
            if (itPixel != union_find_set.end())
                itPixel++;
        }

    // show merged regions image
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Output", output);
}

void scaleSets(const Mat &input, int budget)
{
    map<int, Region> regions; // array of regions
    set<int> activeRegions; // array of regions still present in image
    map<int, Elt> union_find_set; // structure of all regions used for search
    priority_queue<MergeCandidate> priorityQueue;

    // REGIONS INIT

    // create a region for each pixel
    for (int i = 0; i < input.rows; ++i)
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

            Vec3f pixel = input.at<Vec3f>(i, j);

            initESquares(r.eSquares, pixel);
            initEPixels(r.ePixels, pixel);

            regions.emplace(r.id, r);
            activeRegions.emplace(id);
            union_find_set[id] = Elt{1, id};
        }

    // LAMBDAS INIT

    // initial lambda calculation for every region and their neighbors
    for (int regionId: activeRegions)
        for (int neighborIdx: regions[regionId].adjacentRegions)
        {
            double lambda = optimalLambda(regions[regionId], regions[neighborIdx]);
            priorityQueue.emplace(MergeCandidate{
                    regionId,
                    neighborIdx,
                    lambda,
                    4,
                    4
            });
        }

    // MERGING REGIONS

    time_t timeAtLoopStart, timeAfterActiveRegionsPassed, timeAfterRemove, timeAfterEverything,
            totalItTime = 0, totalTimeLoop = 0, totalTimeRemove = 0, totalTimeUpdate = 0;

    double totalNeighbors = 0;
    double lastLambda = -1;
    int count = budget;

    while (activeRegions.size() > 1 && count != 0)
    {
        timeAtLoopStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        Region r1, r2;
        vector<int> doneRegions;

        bool validMerge = false;
        while (!validMerge && !priorityQueue.empty())
        {
            // 2 log n
            int r1Id = priorityQueue.top().r1;
            int r2Id = priorityQueue.top().r2;
            int p1 = priorityQueue.top().perimeter1;
            int p2 = priorityQueue.top().perimeter2;

            lastLambda = priorityQueue.top().lambda;

            r1Id = find_(union_find_set, r1Id);
            r2Id = find_(union_find_set, r2Id);

            priorityQueue.pop();

            if (r1Id != r2Id)
            {
                r1 = regions[r1Id];
                r2 = regions[r2Id];

                // 2 log n
                validMerge = (r1.perimeter == p1 && r2.perimeter == p2) &&
                             (activeRegions.find(r1.id) != activeRegions.end() &&
                              activeRegions.find(r2.id) != activeRegions.end());
            }
            else
            {
                validMerge = false;
            }
        }
        timeAfterActiveRegionsPassed = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        // r2 is merged into r1, only r1 remains
        int newRegionId = r1.id;
        regions[newRegionId] = merge(r1, r2);
        union_(union_find_set, r1.id, r2.id);

        timeAfterRemove = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        // update neighbors of merged region :
        // delete r2 and add r1 in the neighbors list of merged region neighbors
        set<int> neighborsIds;
        Region newR = regions[newRegionId];
        for (int neighborId: regions[newRegionId].adjacentRegions)
        {
            int tNeigId = find_(union_find_set, neighborId);
            if (tNeigId != r2.id && tNeigId != newRegionId && neighborsIds.find(tNeigId) == neighborsIds.end())
            {
                double lambda = optimalLambda(newR, regions[tNeigId]);
                priorityQueue.push(MergeCandidate{
                        newRegionId,
                        tNeigId,
                        lambda,
                        newR.perimeter,
                        regions[tNeigId].perimeter
                });
                neighborsIds.insert(tNeigId);
            }
        }
        regions[newRegionId].adjacentRegions = neighborsIds;
        totalNeighbors += (double) regions[newRegionId].adjacentRegions.size();

        activeRegions.erase(activeRegions.find(r2.id));

        timeAfterEverything = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        totalItTime += (timeAfterEverything - timeAtLoopStart);
        totalTimeLoop += (timeAfterActiveRegionsPassed - timeAtLoopStart);
        totalTimeRemove += (timeAfterRemove - timeAfterActiveRegionsPassed);
        totalTimeUpdate += (timeAfterEverything - timeAfterActiveRegionsPassed);
        if (count % 100 == 0)
        {
            cout << count
                 << " total : " << totalItTime
                 << " loop : " << totalTimeLoop
                 << " remove : " << totalTimeRemove
                 << " rest : " << totalTimeUpdate
                 << " neighbors : " << totalNeighbors
                 << " lambda size : " << priorityQueue.size() << endl;
            totalItTime = 0;
            totalTimeLoop = 0;
            totalTimeUpdate = 0;
            totalNeighbors = 0;
            totalTimeRemove = 0;
        }

        count--;
    }

    displayRegions(input, regions, union_find_set);
}

void waitExit()
{
    while (true)
    {
        int keycode = waitKey(0);
        int asciiCode = keycode & 0xff;
        if (asciiCode == 'q')
            break;
    }
}

int main(int argc, char **argv)
{
    // handling arguments

    argparse::ArgumentParser program("scale set");
    program.add_description("Scale set segmentation algorithm");
    program.add_argument("image")
            .help("path to input image");
    program.add_argument("budget")
            .help("number of maximum merge allowed by user")
            .scan<'i', int>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    auto imagePath = program.get<string>("image");
    auto budget = program.get<int>("budget");

    // program execution

    Mat input = imread(imagePath, IMREAD_COLOR);
    input.convertTo(input, CV_32FC3, 1 / 255.); // convert image to float types

    scaleSets(input, budget);

    namedWindow("Input", WINDOW_AUTOSIZE);
    imshow("Input", input);

    waitExit();

    return 0;
}
