/* Universidad de Chile - FCFM
 * EL5206 - Computational Intelligence Laboratory
 * Final Course Project: Context Based Image Retrieval
 *
 * Author: Sebastian Parra
 * 2018
 */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


/**
 * Enumeration of different methods for region division
 */
enum RegionDivision {RECT9, RECT16, RECT36, NON_RECT};


/**
 * Enumeration of different descriptors used for feature vector calculation (color histogram or edge histogram)
 */
enum Descriptor {DESCR_COLOR, DESCR_EHD};


/**
 * Function to get N equally sized rectangular sub-divisions of an image.
 * N must be a number with a closed square root expression.
 */
std::vector<cv::Mat> getNSubRectangles(cv::Mat sourceHsv, int nRectangles) {
    int top, left;
    auto width = static_cast<int>(sourceHsv.cols / sqrt(nRectangles));
    auto height = static_cast<int>(sourceHsv.rows / sqrt(nRectangles));

    std::vector<cv::Mat> regions;
    for(int i=0; i<nRectangles; i++) {

        left = i % ( (int) sqrt(nRectangles) ) * width;
        top = i / ( (int) sqrt(nRectangles) ) * height;
        regions.push_back(sourceHsv( cv::Rect(left, top, width, height) ));
    }
    return regions;
}

/**
 * Function to determine whether a point (pointRow,pointCol) in a matrix is inside an ellipse with its center
 * on the center of an image of dimensions (rows,cols), with a major axis diameterX, and a minor axis diameterY
 *
 * @param pointRow row index of the point to check
 * @param pointCol column index of the point to check
 * @param rows number of image rows
 * @param cols number of image columns
 * @param diameterX major axis for the ellipse
 * @param diameterY minor axis for the ellipse
 * @return true if point is inside the ellipse, false if otherwise
 */
bool isInsideEllipse(int pointRow, int pointCol, int rows, int cols, int diameterX, int diameterY) {
    double termX = pow(pointCol - cols/2, 2) / pow(diameterX/2, 2);
    double termY = pow(pointRow - rows/2, 2) / pow(diameterY/2, 2);
    return (termX + termY) <= 1;
}

/**
 * Function to divide HSV image in sub-regions from a predetermined set of techniques.
 * For RECT region divisions, this function returns a vector of the rectangular sub-matrices
 * of the original image. Instead, for NON_RECT method, this function returns a mask to be
 * used in histogram calculation
 *
 * @param sourceHsv source HSV image
 * @param reg region division method to use
 * @return vector of Mat object holding all sub-regions
 */
std::vector<cv::Mat> getRegionDivision(cv::Mat sourceHsv, RegionDivision reg) {
    switch(reg) {
        // RECT9: Divide image in 9 equal rectangular regions
        case RECT9:
            return getNSubRectangles(sourceHsv, 9);
        // RECT16: Divide image in 16 equal rectangular regions
        case RECT16:
            return getNSubRectangles(sourceHsv, 16);
        // RECT36: Divide image in 36 equal rectangular regions
        case RECT36:
            return getNSubRectangles(sourceHsv, 36);
        // NON_RECT: Divide image in 5 non rectangular regions
        case NON_RECT: {
            int cols = sourceHsv.cols, rows = sourceHsv.rows;

            cv::Mat topLeft = cv::Mat::zeros(rows, cols, CV_8UC1);
            cv::Mat topRight = cv::Mat::zeros(rows, cols, CV_8UC1);
            cv::Mat bottomLeft = cv::Mat::zeros(rows, cols, CV_8UC1);
            cv::Mat bottomRight = cv::Mat::zeros(rows, cols, CV_8UC1);
            cv::Mat center = cv::Mat::zeros(rows, cols, CV_8UC1);

            int diameterX = cols * 3 / 4, diameterY = rows * 3 / 4;

            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    if(isInsideEllipse(i, j, rows, cols, diameterX, diameterY)) {
                        center.at<uchar>(i, j) = 1;
                    }
                    else if(i < rows/2 && j < cols/2) {
                        //Point is at top left corner and not inside circle
                        topLeft.at<uchar>(i, j) = 1;
                    }
                    else if(i < rows/2) {
                        //Point is at top right corner and not inside circle
                        topRight.at<uchar>(i,j) = 1;
                    }
                    else if(j < cols/2) {
                        //Point is at bottom left corner and not inside circle
                        bottomLeft.at<uchar>(i,j) = 1;
                    } else {
                        bottomRight.at<uchar>(i,j) = 1;
                    }
                }
            }
            return std::vector<cv::Mat> {topLeft, topRight, bottomLeft, bottomRight, center};
        }
    }
}

/**
 * Function to get feature vector of concatenated normalized histograms of every image region
 * (previously calculated)
 *
 * @param regions vector of image region divisions (or vector of masks indicating region locations) obtained
 * via the getRegionDivision function
 * @param reg region division method that was used. Must be RECT9, RECT16, RECT36, or NON_RECT
 * @param sourceHsv source hsv image to process. Only used for NON_RECT division
 * @return feature vector containing the concatenation of each normalized histogram obtained for every region division
 */
cv::Mat getHistogramVector(std::vector<cv::Mat> regions, RegionDivision reg, cv::Mat sourceHsv=cv::Mat()) {
    cv::Mat histVector;
    cv::Mat histH, histS, histV;

    //int channels[] = {0, 1, 2};
    int hChannel[] = {0};
    int sChannel[] = {1};
    int vChannel[] = {2};

    int hBins = 8, sBins = 12, vBins = 3;
    //int histSize[] = {hBins, sBins, vBins};
    int histSizeH[] = {hBins};
    int histSizeS[] = {sBins};
    int histSizeV[] = {vBins};

    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    //const float* ranges[] = { hRanges, sRanges, sRanges};
    const float* rangeH[] = {hRanges};
    const float* rangeS[] = {sRanges};

    for(int i = 0; i < regions.size(); i++) {
        switch(reg) {
            case NON_RECT: {
                cv::calcHist(&sourceHsv, 1, hChannel, regions[i], histH, 1, histSizeH, rangeH, true, false);
                cv::calcHist(&sourceHsv, 1, sChannel, regions[i], histS, 1, histSizeS, rangeS, true, false);
                cv::calcHist(&sourceHsv, 1, vChannel, regions[i], histV, 1, histSizeV, rangeS, true, false);
                break;
            }
            default: {
                cv::calcHist(&regions[i], 1, hChannel, cv::Mat(), histH, 1, histSizeH, rangeH, true, false);
                cv::calcHist(&regions[i], 1, sChannel, cv::Mat(), histS, 1, histSizeS, rangeS, true, false);
                cv::calcHist(&regions[i], 1, vChannel, cv::Mat(), histV, 1, histSizeV, rangeS, true, false);
                break;
            }
        }
        cv::normalize(histH, histH, 255, 0, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(histS, histS, 255, 0, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(histV, histV, 255, 0, cv::NORM_MINMAX, -1, cv::Mat());

        if(i == 0) {
            histVector = histH.t();
        } else {
            cv::hconcat(histVector, histH.t(), histVector);
        }
        cv::hconcat(histVector, histS.t(), histVector);
        cv::hconcat(histVector, histV.t(), histVector);
    }
    return histVector;
}


/**
 * Function to apply 5 different directional edge filters on a sub-block matrix. Used for Edge Histogram Descirptor
 *
 * @param subBlock sub-block matrix to process. Must be of size 2x2
 * @return vector of results obtained with (0) horizontal kernel, (1) vertical kernel, (2) 45°-diagonal kernel,
 * (3) 135°-diagonal kernel, (4) isotropic kernel
 */
std::vector<float > applyEdgeFilters(const cv::Mat &subBlock)
{
    // Calcuate kernels for edge filters
    float horz[2][2] = {{1, -1},{1, -1}};
    float vert[2][2] = {{1, 1}, {-1, -1}};
    float diag45[2][2] = {{static_cast<float>(sqrt(2)), 0}, {0, -1 * static_cast<float>(sqrt(2))}};
    float diag135[2][2] = {{0, static_cast<float>(sqrt(2))}, {-1 * static_cast<float>(sqrt(2)), 0}};
    float iso[2][2] = {{2, -2}, {-2, 2}};

    cv::Mat hKernel = cv::Mat(2, 2, CV_32FC1, horz);
    cv::Mat vKernel = cv::Mat(2, 2, CV_32FC1, vert);
    cv::Mat d45Kernel = cv::Mat(2, 2, CV_32FC1, diag45);
    cv::Mat d135Kernel = cv::Mat(2, 2, CV_32FC1, diag135);
    cv::Mat isoKernel = cv::Mat(2, 2, CV_32FC1, iso);

    // Apply filters to sub-block matrix
    cv::Mat hFiltered, vFiltered, d45Filtered, d135Filtered, isoFiltered;

    cv::filter2D(subBlock, hFiltered, CV_32F, hKernel);
    cv::filter2D(subBlock, vFiltered, CV_32F, vKernel);
    cv::filter2D(subBlock, d45Filtered, CV_32F, d45Kernel);
    cv::filter2D(subBlock, d135Filtered, CV_32F, d135Kernel);
    cv::filter2D(subBlock, isoFiltered, CV_32F, isoKernel);

    // Save results to a vector
    std::vector<float > res;
    res.push_back(hFiltered.at<float>(0, 0));
    res.push_back(vFiltered.at<float>(0, 0));
    res.push_back(d45Filtered.at<float>(0, 0));
    res.push_back(d135Filtered.at<float>(0, 0));
    res.push_back(isoFiltered.at<float>(0, 0));

    return res;
}


/**
 * Function to calculate block matrix of a region of an image for Edge Histogram Descriptor calculation.
 * Each block will have an integer value between 0 and 4 indicating the predominant edge type in the sub-blocks, or 5
 * if there is no such edge.
 *
 * @param region region whose block matrix will be calculated. This shoud be one of the 16 regions obtained on the
 * first step of EHD calculation
 * @param blockRows number of rows that the block matrix will have
 * @param blockCols number of cols that the block matrix will have
 * @param thresh threshold that must be overcome by the predominant edge in order for it to be counted as relevant
 * @return block matrix for the given region, where each index will show what is the predominant edge in that block,
 * if there is one
 */
cv::Mat getBlockMatrix(cv::Mat region, int blockRows, int blockCols, float thresh)
{
    int blockWidth = region.cols / blockCols;
    if(blockWidth % 2 != 0) {blockWidth--;};

    int blockHeight = region.rows / blockRows;
    if(blockHeight % 2 != 0) {blockHeight--;};

    cv::Mat blocks(blockRows, blockCols, CV_16UC1);
    // For each block, compute texture value
    for(int i = 0; i < blockRows; ++i)
    {
        for(int j = 0; j < blockCols; ++j)
        {
            int blockX = i * blockWidth;
            int blockY = j * blockHeight;
            cv::Mat block = region(cv::Rect(blockX, blockY, blockWidth, blockHeight));

            // First separate block in 2x2 average pixels
            cv::Mat subBlock(2, 2, CV_32F, cv::Scalar(0));

            for(int k = 0; k < 4; ++k)
            {
                int subBlockX = (k % 2) * (blockWidth / 2);
                int subBlockY = (k / 2) * (blockHeight / 2);

                cv::Mat mean;
                cv::meanStdDev(block(cv::Rect(subBlockX, subBlockY, blockWidth / 2, blockHeight / 2)),
                                     mean, cv::noArray());

                mean.copyTo(subBlock(cv::Rect(k % 2, k / 2, 1, 1)));
            }

            // Then, apply 5 edge filters to sub block
            std::vector<float > edgeValues = applyEdgeFilters(subBlock);

            // Matrix value at block position will be equal to filter type that maximizes edge filter value, if it is
            // greater than a fixed threshold
            float max = *std::max_element(edgeValues.begin(), edgeValues.end());
            int blockVal;

            if(max > thresh)
            {
                blockVal = static_cast<int>(std::distance(edgeValues.begin(), std::max_element(edgeValues.begin(),
                                                                                               edgeValues.end())));
            } else
            {
                blockVal = 5;
            }

            // Finally, set block value
            blocks.at<short>(i, j) = static_cast<short>(blockVal);
        }
    }
    return blocks;
}


/**
 * Function used to retrieve Edge Histogram Descriptor feature vector of a given image. This function considers the
 * image has been divided en 16 equally sized rectangular regions
 *
 * @param regions vector of regions that compose the original image
 * @param blockRows number of rows to use for the block matrix
 * @param blockCols number of cols to use for the block matrix
 * @return feature vector corresponding to EHD of source image
 */
cv::Mat getTextureVector(std::vector<cv::Mat> regions, int blockRows, int blockCols)
{
    cv::Mat histVector;

    float thresh = 15;
    // For each region, get its matrix of blocks
    for(int i = 0; i < regions.size(); ++i)
    {
        // First, get block matrix
        cv::Mat blocks = getBlockMatrix(regions[i], blockRows, blockCols, thresh);

        // Then, calculate 5-bin histogram
        int bins = 5;
        int histSize[] = {bins};

        float histRange[] = {0, 4};
        const float* ranges[] = { histRange };

        cv::Mat hist;

        int channels[] = {0};

        cv::calcHist(&blocks, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

        // Finally, append to feature vector
        if(i == 0) {
            histVector = hist.t();
        } else {
            cv::hconcat(histVector, hist.t(), histVector);
        }
    }
    return histVector;
}


/**
 * Function to calculate feature vector of a given image. In order to obtain
 * features, the following steps are followed:
 * 1) Image is converted to HSV space
 * 2) Image is divided in regions, using one of the four specified methods to do this operation
 * 3) Histograms are calculated for each region using the following convention:
 *      3.1) 8 bins for H channel
 *      3.2) 12 bins for S channel
 *      3.3) 3 bins for V channel
 * 4) Histograms are normalized and concatenated into a feature vector
 *
 * For region division step, method chosen must be one of RECT9, RECT16, RECT36, or NON_RECT,
 * where the first three divide the image on a certain number of equally-sized rectangles,
 * while NON_RECT uses a circular region at the center of the image, and then divides the
 * rest of the space in 4 equally-sized portions, each holding a corner of the image.
 *
 * @param sourceImage the source BGR image to process
 * @param regionType the type of region division to use for feature vector calculation.
 * Must be one of RECT9, RECT16, RECT36, or NON_RECT. This parameter is ignored when desc is DESCR_EHD
 * @param desc descriptor to use for feature vector calculation. Must be either DESCR_COLOR or DESCR_EHD
 * @return a feature vector with the concatenation of all histograms calculated
 */
cv::Mat getFeatureVector(const cv::Mat &sourceImage, RegionDivision regionType, Descriptor desc)
{
    switch(desc)
    {
        case DESCR_COLOR: {
            // Step 1: Convert source image to HSV space
            cv::Mat sourceHsv;
            cv::cvtColor(sourceImage, sourceHsv, cv::COLOR_BGR2HSV);

            // Step 2: Divide HSV image in sub-regions according to predetermined methods
            std::vector<cv::Mat> regions = getRegionDivision(sourceHsv, regionType);

            // Step 3: Calculate histograms for each region, normalize them, and get feature vector of concatenations
            cv::Mat featureVector;
            if(regionType == NON_RECT) {
                featureVector = getHistogramVector(regions, regionType, sourceHsv);
            } else {
                featureVector = getHistogramVector(regions, regionType, cv::Mat());
            }

            return featureVector;
        }

        case DESCR_EHD: {
            // Step 1: Convert source image to grayscale
            cv::Mat sourceGray;
            cv::cvtColor(sourceImage, sourceGray, cv::COLOR_BGR2GRAY);

            // Step 2: Divide gray image in 4x4 equally sized sub-images
            std::vector<cv::Mat> regions = getRegionDivision(sourceGray, RECT16);

            // Step 3: Divide each region in a matrix of 10x10 blocks
            int blockRows = 15;
            int blockCols = 15;

            cv::Mat featureVector = getTextureVector(regions, blockRows, blockCols);
            return featureVector;
        }
    }

}


int main(int argc, char *argv[])
{
    cv::String keys =
            "{@nRects | 0 | number of rectangles to use for region division (9, 16, 36, or 0 for non-rectangular). Ignored if texture flag is set}"
            "{@dbFolder | ../img_database | image database folder}"
            "{@dbVectFolder | ../database_feat_vector | feature vector database folder}"
            "{@queryFolder | ../img_query | query image folder}"
            "{@queryVectFolder | ../query_feat_vector | feature vector query folder}"
            "{help h ?|     | show help message}"
            "{texture |     | set this flag to use texture-based descriptor (Edge Histogram Descriptor) to calculate feature vectors}";

    cv::CommandLineParser parser(argc, argv, keys);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }

    Descriptor d = DESCR_COLOR;

    RegionDivision reg;

    // Set region division method to use
    auto nRects = parser.get<int>("@nRects");
    if(nRects == 9)
        reg = RECT9;
    else if(nRects == 16)
        reg = RECT16;
    else if(nRects == 36)
        reg = RECT36;
    else if(nRects == 0)
        reg = NON_RECT;
    else
    {
        std::cout << "Error: Invalid number of rectangles. Must be 9, 16, 36, or 0" << std::endl;
        return -2;
    }

    // If "texture" flag is set, calculate EHD descriptor instead (forces region division of 16 rectangles)
    if(parser.has("texture"))
    {
        d = DESCR_EHD;
        std::cout << "Processing images with Edge Histogram Descriptor..." << std::endl;
    } else
    {
        std::cout << "Processing images with Color Histogram Descriptor..." << std::endl;
    }

    // Get vector of all filenames in image database
    cv::String dbFolder = parser.get<cv::String>("@dbFolder");
    std::vector<cv::String> dbPaths;
    cv::glob(dbFolder, dbPaths, true);

    cv::String dbOutPath = parser.get<cv::String>("@dbVectFolder");

    std::cout << "Processing database..." << std::endl;
    for (auto &dbPath : dbPaths)
    {
        std::cout << "Reading database image: " << dbPath << std::endl;
        // For each img in database, read it, calculate feature vector, and save it as .xml
        cv::Mat dbSrc = cv::imread(dbPath);
        if (dbSrc.empty())
        {
            std::cout << "Could not find image! " << dbPath << std::endl;
            return -1;
        }

        cv::Mat dbFeatVector = getFeatureVector(dbSrc, reg, d);

        std::string imgName = dbPath.substr(dbPath.size() - 10, 6   );

        std::stringstream ss;
        ss << dbOutPath << "/" << imgName << ".xml";
        cv::String dbOutFile = ss.str();

        cv::FileStorage fs(dbOutFile, cv::FileStorage::WRITE);

        fs << "featVector" << dbFeatVector;

        fs << "codeName" << imgName;

        fs.release();
    }

    // Get vector of all query image filenames
    cv::String queryFolder = parser.get<cv::String>("@queryFolder");
    std::vector<cv::String> queryPaths;
    cv::glob(queryFolder, queryPaths, true);

    cv::String queryOutPath = parser.get<cv::String>("@queryVectFolder");

    std::cout << "Processing query images..." << std::endl;
    for (auto &queryPath : queryPaths)
    {
        std::cout << "Reading query image: " << queryPath << std::endl;
        // For each img in database, read it, calculate feature vector, and save it as .xml
        cv::Mat querySrc = cv::imread(queryPath);
        if (querySrc.empty())
        {
            std::cout << "Could not find image! " << queryPath << std::endl;
            return -1;
        }

        cv::Mat queryFeatVector = getFeatureVector(querySrc, reg, d);

        std::string imgName = queryPath.substr(queryPath.size() - 10, 6);

        std::stringstream ss;
        ss << queryOutPath << "/" << imgName << ".xml";
        cv::String queryOutFile = ss.str();

        cv::FileStorage fs(queryOutFile, cv::FileStorage::WRITE);

        fs << "featVector" << queryFeatVector;

        fs << "codeName" << imgName;

        fs.release();
    }
}