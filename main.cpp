#include <iostream>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <set>

using namespace std;
using namespace cv;

using Pixel = pair<int, int>;

struct Region
{
    int id{-1}; // TODO remove ?
    set<Pixel> pixels;
    set<int> adjacentRegions; // TODO remove ?
    int perimeter{0};
};

double mean(const Mat &image, Region& region)
{
    double sum;
    for (Pixel pixel : region.pixels)
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
double variance(const Mat& image, Region& region)
{
    double variances[3];

    // color channels split
    Mat channels[3];
    split(image, channels);

    // variance for each channels
    for (int i = 0; i < 3; ++i)
    {
        double sum = 0.;
        double meanChannel = mean(channels[i], region);
        for (Pixel pixel : region.pixels) {
            sum += pow(((double) channels[i].at<float>(pixel.first, pixel.second)) - meanChannel, 2);
        }
        variances[i] = sum / (double) region.pixels.size();
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
int mergedPerimeter(const Region& region1, const Region& region2)
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
    for (Pixel pixel : smallRegion.pixels) // n
    {
        vector<Pixel> neighbors = getPixelNeighbors(pixel);
        for (auto neighbor : neighbors) // 4n
            if (bigRegion.pixels.find(neighbor) != bigRegion.pixels.end()) // log(n)
                length++;
    }
    return length;
}

double energy(const Mat& image, Region& region, double lambda)
{
    return variance(image, region) + lambda * region.perimeter;
}

Region simulatedMerge(Region* region1, Region* region2)
{
    Region outR1;

    // TODO copy pixels and adjacent regions to avoid side effect
    // copy r1
    outR1 = *region1;
    //outR1.perimeter = region1->perimeter;

    // update r1
    outR1.perimeter = outR1.perimeter + region2->perimeter - mergedPerimeter(outR1, *region2); // update perimeter

    outR1.pixels.merge(region2->pixels); // merge pixels
    outR1.adjacentRegions.merge(region2->adjacentRegions); // add new adjacent regions


    return outR1;
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
Region merge(Region region1, Region region2, vector<vector<int>>* regionIds)
{
    // merge region2 into region1
    Region mergedRegion = region1;
    mergedRegion.pixels.merge(region2.pixels); // merge pixels
    mergedRegion.adjacentRegions.merge(region2.adjacentRegions); // add new adjacent regions
    mergedRegion.adjacentRegions.erase(find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region2.id));
    mergedRegion.adjacentRegions.erase(find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region1.id));

    // switch region ids to region1's for pixels which were in region2
    for (auto & regionId : *regionIds)
        for (int & id : regionId)
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
 * @param image
 * @return optimal lambda value
 */
double optimalLambda(Region& region1, Region& region2, const Mat& image)
{
    double variance1 = variance(image, region1);
    double variance2 = variance(image, region2);

    double perimeter1 = region1.perimeter;
    double perimeter2 = region2.perimeter;

    // merge region2 into region1
    Region mergedRegion = region1;
    set<Pixel> region2Pixels = region2.pixels;
    set<int> region2AdjacentRegions = region2.adjacentRegions;
    mergedRegion.pixels.merge(region2Pixels); // merge pixels
    mergedRegion.adjacentRegions.merge(region2AdjacentRegions); // add new adjacent regions

    mergedRegion.adjacentRegions.erase(find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region2.id));
    mergedRegion.adjacentRegions.erase(find(mergedRegion.adjacentRegions.begin(), mergedRegion.adjacentRegions.end(), region1.id));

    double variance1U2 = variance(image, mergedRegion);
    double perimeter1U2 = mergedRegion.perimeter;

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
void displayRegions(const Mat& input, vector<Region> regions, vector<vector<int>> regionIds)
{
    Mat output;
    input.copyTo(output);

    vector<int> encounteredRegions;
    vector<Vec3b> encounteredRegionsColors;

    // coloring regions in output with a new random color for each region
    for (int i = 0; i < input.rows; ++i)
        for (int j = 0; j < input.cols; ++j)
        {
            int regionId = regions[regionIds.at(i).at(j)].id;

            // check if region has already been encountered, if not, create a new color for it
            if (find(encounteredRegions.begin(), encounteredRegions.end(), regionId) == encounteredRegions.end())
            {
                encounteredRegions.push_back(regionId);
                encounteredRegionsColors.emplace_back(rand() % 255, rand() % 255, rand() % 255);
            }

            // paint output with corresponding color
            output.at<Vec3f>(i, j) = encounteredRegionsColors[regionId];
        }

    // show merged regions image
    namedWindow("Merged regions", WINDOW_AUTOSIZE);
    imshow("Merged regions", output);
}

void scaleSets(Mat input)
{
    // create two fake init regions
    vector<Region> regions; // array of regions
    vector<int> activeRegions; // array of regions present in image
    regions.push_back(Region{
            0,
            set<Pixel>(),
            set<int>{1, 2},
            4
    });
    activeRegions.push_back(0);

    regions.push_back(Region{
            1,
            set<Pixel>(),
            set<int>{0,3},
            4
    });
    activeRegions.push_back(1);

    regions.push_back(Region{
            2,
            set<Pixel>(),
            set<int>{0,3},
            4
    });
    activeRegions.push_back(2);

    regions.push_back(Region{
            3,
            set<Pixel>(),
            set<int>{2,1},
            4
    });
    activeRegions.push_back(3);//*/

    // fill fake regions with their pixels (they take half of the image)
    // and mark their pixels as part of their region in regionIds
    vector<vector<int>> regionIds;
    for (int i = 0; i < input.rows; ++i)
    {
        regionIds.push_back(vector<int>());
        for (int j = 0; j < input.cols; ++j)
        {
            // partition pixel between regions
            if (j < 75 && i < 50)
            {
                regions[0].pixels.emplace(Pixel(i, j));
                regionIds[i].push_back(0);// i * input.cols + j;
            }
            else if (j >= 75 && i < 50)
            {
                regions[1].pixels.emplace(Pixel(i, j));
                regionIds[i].push_back(1);
            }
            else if (j < 75 && i >= 50)
            {
                regions[2].pixels.emplace(Pixel(i, j));
                regionIds[i].push_back(2);
            }
            else
            {
                regions[3].pixels.emplace(Pixel(i, j));
                regionIds[i].push_back(3);
            }

            /*regions.push_back(Region{
                    i * input.cols + j,
                    set<Pixel>{Pixel(i, j)},
                    getInitRegionNeighbors(Pixel(i, j), input.cols, input.rows),
                    4
            });
            activeRegions.push_back(i * input.cols + j);//*/
        }
    }
    //displayRegions(input, regions, regionIds);
    int nbCount = 5;
    int count = nbCount;
    while (activeRegions.size() > 1 && count != 0)
    {
        cout << "nbRegions " << activeRegions.size() << endl;
        Region r1, r2;
        double lambdaMin = 100000000.;
        // TODO : handle 'active' regions and optimise code (obj don't calculate the same distance twice)
        vector<int> doneRegions;

        // find minimal lambda and the merge region associated with it
        // loop through all present regions
        for (int regionId : activeRegions)
        {
            for (int neighborIdx: regions[regionId].adjacentRegions)
            {
                // neighbor is not in doneRegions
                // TODO : update indexs (problem : if neighbor isn't active anymore)
                if (find(doneRegions.begin(), doneRegions.end(), neighborIdx) == doneRegions.end())
                {
                    double lambda = optimalLambda(regions[regionId], regions[neighborIdx], input);
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
        cout << "merge " << r1.id << " " << r2.id << endl;
        if (r1.id == r2.id) {
            cout << "ERROR merge with itself" << endl;
        }
        regions[r1.id] = merge(r1,r2, &regionIds);

        activeRegions.erase(find(activeRegions.begin(), activeRegions.end(), r2.id));

        //if (count % (nbCount/100) == 100)
            //cout << "Count : " << count << endl;
        count--;
    }

    displayRegions(input, regions, regionIds);//*/
}


int main()
{
    Mat input = imread("../images/pobo_small.png", IMREAD_COLOR);
    input.convertTo(input, CV_32FC3, 1.0/255.0); // convert image to float type

    // show original image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input);

    scaleSets(input);

    Mat output;
    input.copyTo(output);

    // show output image
    //namedWindow("Output", WINDOW_AUTOSIZE);
    //imshow("Output", output);
    while (true) {
        int keycode = waitKey( 50 );
        int asciicode = keycode & 0xff;
        if (asciicode == 'q')
            break;
    }
    return 0;
}
