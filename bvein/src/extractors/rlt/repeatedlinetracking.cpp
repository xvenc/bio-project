#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

// Export as C so we can call directly and no name mangling is performed
extern "C" {

    /** Loads image and mask from files and returns them as cv::Mat instances */
    void ReadInputFiles(char* image_path, char* mask_path, cv::OutputArray image, cv::OutputArray mask) {
        cv::Mat image_loaded = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat mask_loaded = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);

        if (image_loaded.empty() || mask_loaded.empty()) {
            std::cerr << "Error: Unable to load image or mask file!" << std::endl;
            return;
        }

        image.create(image_loaded.size(), CV_8U);
        mask.create(mask_loaded.size(), CV_8U);

        image_loaded.copyTo(image);
        mask_loaded.copyTo(mask);
    }

    /** Saves the extracted vein image to a file */
    void SaveExtracted(char* out_path, cv::InputArray image) {
        cv::imwrite(out_path, image);
    }

    /** Implementation of the Repeated Line Tracking algorithm */
    void RepeatedLineTracking(char* image_path, char* mask_path, char* out_path, unsigned iterations, unsigned r, unsigned W) {
        cv::Mat src, mask;
        ReadInputFiles(image_path, mask_path, src, mask);

        // Convert source image to float64 and normalize
        src.convertTo(src, CV_64F, 1.0 / 255.0);

        // Probability of moving left/right and up/down
        double p_lr = 0.5;
        double p_ud = 0.25;

        // Locus space for tracking information
        cv::Mat Tr = cv::Mat::zeros(src.size(), CV_8U);

        // Directions for neighbor pixels (8 directions plus center)
        cv::Mat bla = (cv::Mat_<char>(9, 2) <<
            -1, -1,
            -1, 0,
            -1, 1,
             0, -1,
             0, 0,
             0, 1,
             1, -1,
             1, 0,
             1, 1);

        // Ensure W is odd
        if (W % 2 == 0) {
            std::cerr << "FAIL: W must be odd!" << std::endl;
            throw std::exception();
        }

        // Calculate r and W-based values for oblique and horizontal/vertical directions
        int ro = cvRound(r * std::sqrt(2) / 2);
        int hW = (W - 1) / 2;
        int hWo = cvRound(hW * std::sqrt(2) / 2);

        // Exclude unreachable regions near the borders
        for (int x = 0; x < src.cols; x++) {
            for (int y = 0; y <= r + hW; y++) {
                mask.at<uchar>(y, x) = 0;                   // Exclude top border
                mask.at<uchar>(src.rows - y - 1, x) = 0;    // Exclude bottom border
            }
        }
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x <= r + hW; x++) {
                mask.at<uchar>(y, x) = 0;                   // Exclude left border
                mask.at<uchar>(y, src.cols - x - 1) = 0;    // Exclude right border
            }
        }

        // RNG for random decisions
        cv::RNG rng;

        // Generate uniformly distributed starting points inside the mask
        std::vector<cv::Point> indices;
        for (int i = 0; i < iterations;) {
            int xRandom = rng.uniform(0, mask.cols);
            int yRandom = rng.uniform(0, mask.rows);
            cv::Point p(xRandom, yRandom);
            if (mask.at<uchar>(p)) {
                indices.push_back(p);
                i++;
            }
        }

        // Main loop: Process each starting point
        for (auto& startingPoint : indices) {
            int xc = startingPoint.x;  // Current x-coordinate
            int yc = startingPoint.y;  // Current y-coordinate

            // Randomly determine movement directions (left/right, up/down)
            int Dlr = rng.uniform(0, 2) ? 1 : -1;
            int Dud = rng.uniform(0, 2) ? 1 : -1;

            // Create the locus-position tracking matrix for this point
            cv::Mat Tc = cv::Mat::zeros(src.size(), CV_8U);

            double Vl = 1;
            while (Vl > 0) {
                // Define candidate neighborhood points (3x3 grid)
                cv::Mat Nr = cv::Mat::zeros(cv::Size(3, 3), CV_8U);

                double random = rng.uniform(0, 101) / 100.0;
                if (random < p_lr) {
                    // Move left or right
                    Nr.at<uchar>(cv::Point(1 + Dlr, 0)) = 1;
                    Nr.at<uchar>(cv::Point(1 + Dlr, 1)) = 1;
                    Nr.at<uchar>(cv::Point(1 + Dlr, 2)) = 1;
                } else if (random >= p_lr && random < (p_lr + p_ud)) {
                    // Move up or down
                    Nr.at<uchar>(cv::Point(0, 1 + Dud)) = 1;
                    Nr.at<uchar>(cv::Point(1, 1 + Dud)) = 1;
                    Nr.at<uchar>(cv::Point(2, 1 + Dud)) = 1;
                } else {
                    // Move in any direction
                    Nr = cv::Mat::ones(cv::Size(3, 3), CV_8U);
                    Nr.at<uchar>(cv::Point(1, 1)) = 0;
                }

                // Determine valid neighboring points
                std::vector<cv::Point> Nc;
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int x = xc + dx;
                        int y = yc + dy;
                        if ((!Tc.at<uchar>(cv::Point(x, y))) && Nr.at<uchar>(cv::Point(dx + 1, dy + 1)) && mask.at<uchar>(cv::Point(x, y))) {
                            int tmp = (dx + 1) * 3 + (dy + 1);
                            Nc.push_back(cv::Point(xc + bla.at<char>(cv::Point(0, tmp)), yc + bla.at<char>(cv::Point(1, tmp))));
                        }
                    }
                }

                // If no valid neighbors, stop tracking this point
                if (Nc.empty()) {
                    Vl = -1;
                    continue;
                }

                // Detect dark line direction near the current tracking point
                std::vector<double> Vdepths(Nc.size());
                for (size_t i = 0; i < Nc.size(); ++i) {
                    cv::Point Ncp = Nc[i];
                    int xp, yp;

                    if (Ncp.y == yc) {
                        yp = Ncp.y;
                        xp = (Ncp.x > xc) ? Ncp.x + r : Ncp.x - r;
                        Vdepths[i] = src.at<double>(cv::Point(xp, yp + hW)) -
                                     2 * src.at<double>(cv::Point(xp, yp)) +
                                     src.at<double>(cv::Point(xp, yp - hW));
                    } else if (Ncp.x == xc) {
                        xp = Ncp.x;
                        yp = (Ncp.y > yc) ? Ncp.y + r : Ncp.y - r;
                        Vdepths[i] = src.at<double>(cv::Point(xp + hW, yp)) -
                                     2 * src.at<double>(cv::Point(xp, yp)) +
                                     src.at<double>(cv::Point(xp - hW, yp));
                    } else if ((Ncp.x > xc && Ncp.y < yc) || (Ncp.x < xc && Ncp.y > yc)) {
                        xp = (Ncp.x > xc) ? Ncp.x + ro : Ncp.x - ro;
                        yp = (Ncp.y < yc) ? Ncp.y - ro : Ncp.y + ro;
                        Vdepths[i] = src.at<double>(cv::Point(xp - hWo, yp - hWo)) -
                                     2 * src.at<double>(cv::Point(xp, yp)) +
                                     src.at<double>(cv::Point(xp + hWo, yp + hWo));
                    } else {
                        xp = (Ncp.x < xc) ? Ncp.x - ro : Ncp.x + ro;
                        yp = (Ncp.y < yc) ? Ncp.y - ro : Ncp.y + ro;
                        Vdepths[i] = src.at<double>(cv::Point(xp - hWo, yp + hWo)) -
                                     2 * src.at<double>(cv::Point(xp, yp)) +
                                     src.at<double>(cv::Point(xp + hWo, yp - hWo));
                    }
                }

                // Mark current position as visited
                Tc.at<uchar>(cv::Point(xc, yc)) = true;
                Tr.at<uchar>(cv::Point(xc, yc))++;

                // Move to the best candidate (deepest valley)
                size_t index = std::distance(Vdepths.begin(), std::max_element(Vdepths.begin(), Vdepths.end()));
                xc = Nc[index].x;
                yc = Nc[index].y;
            }
        }

        SaveExtracted(out_path, Tr);
    }
}