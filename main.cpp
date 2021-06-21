/* Universidad de Chile - FCFM
 * EL5206 - Computational Intelligence Laboratory
 * Final Course Project: Context Based Image Retrieval
 *
 * Author: Sebastián Parra
 * 2018
 */

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <jmorecfg.h>


/**
 * Function to compare distances between (label, distance) pairs
 *
 * @param i first (label, distance) pair to compare
 * @param j second (label,distance) pair to compare
 * @return true if first pair's distance is less than second pair's, or false if otherwise
 */
bool compareDistance(std::pair<cv::String, double> i, std::pair<cv::String, double> j)
{
    return i.second < j.second;
}


int main(int argc, char *argv[])
{
    cv::String keys =
            "{@queryVectFolder | ../query_feat_vector | query feature vector folder}" //query_feat_vector/100500.xml
            "{@dbVectFolder | ../database_feat_vector | database feature vector folder}"
            "{@queryImgFolder | ../img_query | query images folder}"
            "{@dbImgFolder | ../img_database | database images folder}"
            "{showResults |  | show first 10 result images for every query image consulted, by order of relevance}"
            "{help h ?|  | display help message}";

    cv::CommandLineParser parser(argc, argv, keys);
    if(parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }

    // Get vector of all query feature vector filenames
    cv::String queryFolder = parser.get<cv::String>("@queryVectFolder");
    std::vector<cv::String> queryPaths;
    cv::glob(queryFolder, queryPaths, true);

    // Get vector of all database feature vector filenames
    cv::String dbFolder = parser.get<cv::String>("@dbVectFolder");
    std::vector<cv::String> dbPaths;
    cv::glob(dbFolder, dbPaths, true);

    // Get database feature vectors and class labels
    std::vector<cv::Mat> database;
    std::vector<cv::String> dbCodeNames;
    for (const auto &dbPath : dbPaths) {
        cv::FileStorage fs;
        fs.open(dbPath, cv::FileStorage::READ);

        if(!fs.isOpened())
        {
            std::cout << "Error: Failed to open file " << dbPath << std::endl;
            return -1;
        }

        cv::Mat databaseVect;
        fs["featVector"] >> databaseVect;

        cv::String dbCodeName;
        fs["codeName"] >> dbCodeName;

        database.push_back(databaseVect);
        dbCodeNames.push_back(dbCodeName);
    }

    std::vector<double > ranks;
    std::vector<double > normRanks;
    // First, get feature vector of each query img
    for ( const auto &queryPath : queryPaths)
    {
        cv::FileStorage fs;
        fs.open(queryPath, cv::FileStorage::READ);

        if(!fs.isOpened())
        {
            std::cout << "Error: Failed to open file " << queryPath << std::endl;
            return -1;
        }

        cv::Mat queryVect;
        fs["featVector"] >> queryVect;

        cv::String queryCodeName;
        fs["codeName"] >> queryCodeName;

        std::vector<std::pair<cv::String, double > > distances;
        // Compare every query img with all database feature vectors and create vector of distances
        for ( int i = 0; i < database.size(); ++i)
        {
            double dist = cv::compareHist(queryVect, database[i], cv::HISTCMP_CHISQR_ALT    );
            distances.emplace_back(dbCodeNames[i], dist);
        }

        // Sort distances in ascending order
        std::sort(std::begin(distances), std::end(distances), compareDistance);

        // Calculate rank and normalized rank
        int rankSum = 0;
        int nRelevant = 0;
        for( int j = 0; j < distances.size(); ++j)
        {
            int dbLabel = std::stoi(distances[j].first.substr(1, 3), nullptr, 10);
            int queryLabel = std::stoi(queryCodeName.substr(1, 3), nullptr, 10);
            // Only sum to rank if image is relevant
            if(dbLabel == queryLabel)
            {
                nRelevant++;
                rankSum += j+1;
            }
        }
        double rank = (double) rankSum / nRelevant;
        double normRank = (rankSum - ( (double) nRelevant * (nRelevant + 1) ) / 2 ) / (nRelevant * database.size());

        ranks.push_back(rank);
        normRanks.push_back(normRank);


        // If the user wants to, show first 10 result images
        if(parser.has("showResults"))
        {
            // First read query image
            std::string queryImgName = queryPath.substr(queryPath.size() - 10, 6);
            cv::String queryImgFolder = parser.get<cv::String>("@queryImgFolder");
            cv::String dbImgFolder = parser.get<cv::String>("@dbImgFolder");

            std::stringstream ss;
            ss << queryImgFolder << "/" << queryImgName << ".jpg";
            cv::String queryImgPath = ss.str();

            cv::Mat queryImg = cv::imread(queryImgPath);
            if(queryImg.empty())
            {
                std::cout << "Could not find image! " << queryImgPath << std::endl;
                return -2;
            }

            for(int i = 0; i < 10; ++i)
            {
                // Then, for each of the 10 first results, show a composite image with the query on the left, and the
                // result on the right
                std::string dbImgName = distances[i].first;

                std::stringstream ssDatabase;
                ssDatabase << dbImgFolder << "/" << dbImgName << ".jpg";
                cv::String dbImgPath = ssDatabase.str();

                cv::Mat dbImg = cv::imread(dbImgPath);
                if(dbImg.empty())
                {
                    std::cout << "Could not find image! " << queryImgPath << std::endl;
                    return -2;
                }

                int joinedRows = cv::max(queryImg.rows, dbImg.rows);
                int joinedCols = queryImg.cols + dbImg.cols;
                cv::Mat3b joined(joinedRows, joinedCols, cv::Vec3b(0, 0, 0));

                queryImg.copyTo(joined(cv::Rect(0, 0, queryImg.cols, queryImg.rows)));
                dbImg.copyTo(joined(cv::Rect(queryImg.cols, 0, dbImg.cols, dbImg.rows)));

                std::stringstream ssWindow;
                ssWindow << "Query image: " << queryImgName << ".jpg (left) \t" << "Result number " << (i+1) << ": " <<
                        dbImgName << ".jpg (right)";
                cv::String winName = ssWindow.str();

                cv::namedWindow(winName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
                cv::resizeWindow(winName, 1500, 700);
                cv::imshow(winName, joined);

                cv::waitKey();
            }

        }
    }

    double avgRank = std::accumulate(ranks.begin(), ranks.end(), 0.0) / ranks.size();
    double avgNormRank = std::accumulate(normRanks.begin(), normRanks.end(), 0.0) / normRanks.size();

    std::cout << "Average rank for all query images: " << avgRank << std::endl;
    std::cout << "Average normalized rank for all query images: " << avgNormRank << std::endl;
}

