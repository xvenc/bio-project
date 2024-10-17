#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/operations.hpp>
#include <algorithm>

// Export as C so we can call directly and no name mangling is performed
extern "C" {
	void ReadInputFiles(char* image_path, char* mask_path, cv::OutputArray image, cv::OutputArray mask) {
		/** Loads image and mask from files and returns them as cv::Mat instances */
		cv::Mat image_loaded = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
		cv::Mat mask_loaded = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);

		image.create(image_loaded.size(), CV_8U);
		mask.create(mask.size(), CV_8U);

		image_loaded.copyTo(image);
		mask_loaded.copyTo(mask);
	}

	void SaveExtracted(char* out_path, cv::InputArray image) {
		cv::imwrite(out_path, image);
	}

	void RepeatedLineTracking(char* image_path, char* mask_path, char* out_path, unsigned iterations, unsigned r, unsigned W) {
		cv::Mat src, mask;
		ReadInputFiles(image_path, mask_path, src, mask);

		// TODO: implement RLT, currently the original image is only copied to Tr (locus space) matrix

		cv::Mat Tr;
		src.copyTo(Tr);

		// Here the Tr should contain the extracted image

		// Write file to given location
		SaveExtracted(out_path, Tr);
	}
}
