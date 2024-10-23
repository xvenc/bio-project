/**
 * @brief Repeated Line Tracking (RLT) algorithm implementation
 * @file repeatedlinetracking.cpp
 * @details This implementation was inspired by https://www.mathworks.com/matlabcentral/fileexchange/35716-miura-et-al-vein-extraction-methods
 */

#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

// Random number generator
cv::RNG rng;

// Probability of moving left/right and up/down
int PLR = 50;
int PUD = 25;

// Directions for neighbor pixels (8 directions plus center)
char _neigh[9][2] = {
    {-1, -1}, {-1, 0}, {-1, 1},
    {0, -1}, {0, 0}, {0, 1},
    {1, -1}, {1, 0}, {1, 1}
};

// Matrix for neighbor pixel positions
cv::Mat neigh_positions(9, 2, CV_8S, _neigh);

// Struct to hold current position and movement directions
struct Positions {
    unsigned x;
    unsigned y;
    int Dlr;
    int Dud;
};

// Parameters for the cross-section profile
struct ProfileValues {
    unsigned r;
    unsigned ro;
    unsigned W;
    unsigned hW;
    unsigned hWo;
};

/** Loads image and mask from file paths and returns them as matrix instances
 * @param image_path Path to the image file
 * @param mask_path Path to the mask file
 * @param image Output image matrix
 * @param mask Output mask matrix
 */
void ReadInputFiles(const char* image_path, const char* mask_path, cv::OutputArray image, cv::OutputArray mask) {
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

/**
 * Saves the extracted image to the specified path
 * @param out_path Path to save the extracted image
 */
void SaveExtracted(const char* out_path, const cv::InputArray image) {
    cv::imwrite(out_path, image);
}

/** Exclude unreachable regions near the borders of the mask
 * @param mask Mask matrix
 * @param p Profile values
 */
void ShrinkMask(cv::Mat &mask, struct ProfileValues p) {
    cv::Range cols(0, mask.cols);
    cv::Range rows(0, mask.rows);

    cv::Range leftCols(0, p.r + p.hW + 1);
    cv::Range rightCols(mask.cols - (p.r + p.hW + 1), mask.cols);
    cv::Range topRows(0, p.r + p.hW + 1);
    cv::Range bottomRows(mask.rows - (p.r + p.hW + 1), mask.rows);

    mask(rows, leftCols) = 0;
    mask(rows, rightCols) = 0;
    mask(topRows, cols) = 0;
    mask(bottomRows, cols) = 0;
}

/** Generates starting points within the mask for given number of iterations
 * @param mask Mask matrix
 * @param iterations Number of starting points to generate
 */
std::vector<cv::Point> GenerateStartingPoints(const cv::Mat &mask, const unsigned iterations) {
    std::vector<cv::Point> validPoints;

    // Collect all valid points from the mask
    for (unsigned y = 0; y < mask.rows; ++y) {
        for (unsigned x = 0; x < mask.cols; ++x) {
            if (mask.at<unsigned char>(y, x) != 0) {
                validPoints.emplace_back(x, y);
            }
        }
    }

    // Randomly select points from the valid points of the mask
    std::vector<cv::Point> points;
    for (unsigned i = 0; i < iterations; ++i) {
        int index = rng.uniform(0, static_cast<int>(validPoints.size()));
        points.push_back(validPoints[index]);
    }
    return points;
}

/**
 * Get the neighborhood of the current point
 * @param Tc Current locus position tracking matrix
 * @param mask Mask matrix
 * @param p Current position and movement directions
 */
std::vector<cv::Point> GetNeighborhood(const cv::Mat &Tc, const cv::Mat &mask, const struct Positions &p) {
    // Define candidate neighborhood points (3x3 grid) - depends on direction
    cv::Mat directedNeighborhood = cv::Mat::zeros(cv::Size(3, 3), CV_8U);

    unsigned direction = rng.uniform(0, 101);
    if (direction < PLR) {
        // Move left or right
        directedNeighborhood(cv::Rect(1 + p.Dlr, 0, 1, 3)) = 1;
    } else if (direction >= PLR && direction < (PLR + PUD)) {
        // Move up or down
        directedNeighborhood(cv::Rect(0, 1 + p.Dud, 3, 1)) = 1;
    } else {
        // Move in any direction
        directedNeighborhood = cv::Mat::ones(cv::Size(3, 3), CV_8U);
        directedNeighborhood.at<unsigned char>(cv::Point(1, 1)) = 0;
    }

    // Determine valid neighboring points
    std::vector<cv::Point> Nc;

    // Iterate over the 3x3 neighborhood grid aroud current point
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            // Neighboring point is not in this direction
            if (directedNeighborhood.at<unsigned char>(cv::Point(dx + 1, dy + 1)) == 0) {
                continue;
            }

            // Get the coordinates of the neighboring point in the image
            unsigned x = p.x + dx;
            unsigned y = p.y + dy;

            // If the actual point on the image is outside mask or already visited, skip
            if (mask.at<unsigned char>(cv::Point(x, y)) == 0 || Tc.at<unsigned char>(cv::Point(x, y)) != 0) {
                continue;
            }

            // Get the index of the neighbor pixel and get its push its coordinates into the vector
            unsigned neigh_idx = (dx + 1) * 3 + (dy + 1);
            int x_direction = neigh_positions.at<char>(cv::Point(0, neigh_idx));
            int y_direction = neigh_positions.at<char>(cv::Point(1, neigh_idx));
            Nc.push_back(cv::Point(p.x + x_direction, p.y + y_direction));
        }
    }

    return Nc;
}

std::vector<double> NighCrossProfiles(const cv::Mat &image, const std::vector<cv::Point> &Nc, const struct Positions &pos, const struct ProfileValues &p) {
    unsigned r = p.r;
    unsigned ro = p.ro;
    unsigned W = p.W;
    unsigned hW = p.hW;
    unsigned hWo = p.hWo;
    unsigned xc = pos.x;
    unsigned yc = pos.y;

    std::vector<double> Vdepths(Nc.size());

    // Calculate the darkness of the neighbors
    for (size_t i = 0; i < Nc.size(); ++i) {
        cv::Point N = Nc[i];
        int x, y;

        if (N.y == yc) {
            // Horizontal direction
            y = N.y;
            x = (N.x > xc) ? N.x + r : N.x - r; // Move right or left
            Vdepths[i] = image.at<double>(cv::Point(x, y + hW)) -
                            2 * image.at<double>(cv::Point(x, y)) +
                            image.at<double>(cv::Point(x, y - hW));
        } else if (N.x == xc) {
            // Vertical direction
            x = N.x;
            y = (N.y > yc) ? N.y + r : N.y - r; // Move up or down
            Vdepths[i] = image.at<double>(cv::Point(x + hW, y)) -
                            2 * image.at<double>(cv::Point(x, y)) +
                            image.at<double>(cv::Point(x - hW, y));
        } else if ((N.x > xc && N.y < yc) || (N.x < xc && N.y > yc)) {
            // Diagonal directions
            x = (N.x > xc) ? N.x + ro : N.x - ro; // Move right or left
            y = (N.y < yc) ? N.y - ro : N.y + ro; // Move up or down
            Vdepths[i] = image.at<double>(cv::Point(x - hWo, y - hWo)) -
                            2 * image.at<double>(cv::Point(x, y)) +
                            image.at<double>(cv::Point(x + hWo, y + hWo));
        } else {
            // Diagonal directions (other way)
            x = (N.x < xc) ? N.x - ro : N.x + ro; // Move right or left
            y = (N.y < yc) ? N.y - ro : N.y + ro; // Move up or down
            Vdepths[i] = image.at<double>(cv::Point(x - hWo, y + hWo)) -
                            2 * image.at<double>(cv::Point(x, y)) +
                            image.at<double>(cv::Point(x + hWo, y - hWo));
        }
    }

    return Vdepths;
}

// Export as C so we can call directly and no name mangling is performed
extern "C" {
    /** Implementation of the Repeated Line Tracking algorithm */
    void RepeatedLineTracking(char* image_path, char* mask_path, char* out_path, unsigned iterations, unsigned r, unsigned W) {
        cv::Mat image, mask;
        ReadInputFiles(image_path, mask_path, image, mask);

        // Convert source image to float64
        image.convertTo(image, CV_64F);

        // Locus space for tracking information
        cv::Mat Tr = cv::Mat::zeros(image.size(), CV_8U);

        // Calculate r and W-based values for oblique and horizontal/vertical directions
        const unsigned ro = std::round(r * std::sqrt(2) / 2);
        const unsigned hW = (W - 1) / 2;
        const unsigned hWo = std::round(hW * std::sqrt(2) / 2);
        struct ProfileValues profileValues = {r, ro, W, hW, hWo};

        // Shrink mask to exclude unreachable regions near the borders
        ShrinkMask(mask, profileValues);

        // Generate uniformly distributed starting points inside the mask
        std::vector<cv::Point> startingPoints = GenerateStartingPoints(mask, iterations);

        // Main loop: Process each starting point
        for (auto& startingPoint : startingPoints) {
            unsigned xc = startingPoint.x;  // Current x-coordinate
            unsigned yc = startingPoint.y;  // Current y-coordinate

            // Randomly determine movement directions (left/right, up/down)
            int Dlr = rng.uniform(0, 2) ? 1 : -1;
            int Dud = rng.uniform(0, 2) ? 1 : -1;

            // Create the locus-position tracking matrix for this point
            cv::Mat Tc = cv::Mat::zeros(image.size(), CV_8U);

            double Vl = 1;
            while (Vl > 0) {
                // Get valid neighbors
                struct Positions pos = {xc, yc, Dlr, Dud};
                std::vector<cv::Point> Nc = GetNeighborhood(Tc, mask, pos);

                // If no valid neighbors, stop tracking this point
                if (Nc.empty()) {
                    Vl = -1;
                    continue;
                }

                // Calculate depth of cross-section profiles of the neighbors
                std::vector<double> Vdepths = NighCrossProfiles(image, Nc, pos, profileValues);

                // Mark current position as visited
                Tc.at<unsigned char>(cv::Point(xc, yc)) = true;
                Tr.at<unsigned char>(cv::Point(xc, yc))++;

                // Move to the best candidate (deepest valley)
                size_t best = std::distance(Vdepths.begin(), std::max_element(Vdepths.begin(), Vdepths.end()));
                xc = Nc[best].x;
                yc = Nc[best].y;
            }
        }

        // Save the extracted image without binarization
        SaveExtracted(out_path, Tr);
    }
}