#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include "json.h"
#include "modelHandler.hpp"
#include "convertRoutine.hpp"
#include <time.h>

bool superres(cv::Mat input, cv::Mat& output, float scale, bool noise_reduction) {
	// noise reduction
	if (noise_reduction) {
		std::string modelFileName = "models/noise1_model.json";
		std::vector<w2xc::Model> models;
		if (!w2xc::Model::generateModelFromJSON(modelFileName, models)) {
			return false;
		}

		std::vector<cv::Mat> imageSplit;
		cv::Mat imageY;
		cv::split(input, imageSplit);
		imageSplit[0].copyTo(imageY);

		w2xc::convertWithModels(imageY, imageSplit[0], models);

		cv::merge(imageSplit, input);
	}

	// scaling
	{
		// calculate iteration times of 2x scaling and shrink ratio which will use at last
		int iterTimesTwiceScaling = std::ceil(std::log2(scale));
		double shrinkRatio = 0.0;
		if ((int32_t)scale != std::pow(2, iterTimesTwiceScaling)) {
			shrinkRatio = scale	/ std::pow(2.0, iterTimesTwiceScaling);
		}

		std::string modelFileName = "models/scale2.0x_model.json";
		std::vector<w2xc::Model> models;

		if (!w2xc::Model::generateModelFromJSON(modelFileName, models)) {
			return false;
		}

		std::cout << "start scaling" << std::endl;

		// 2x scaling
		for (int nIteration = 0; nIteration < iterTimesTwiceScaling; nIteration++) {
			std::cout << "#" << std::to_string(nIteration + 1) << " 2x scaling..." << std::endl;

			cv::Size imageSize = input.size();
			imageSize.width *= 2;
			imageSize.height *= 2;
			cv::Mat image2xNearest;
			cv::resize(input, image2xNearest, imageSize, 0, 0, cv::INTER_NEAREST);
			std::vector<cv::Mat> imageSplit;
			cv::Mat imageY;
			cv::split(image2xNearest, imageSplit);
			imageSplit[0].copyTo(imageY);

			// generate bicubic scaled image and split
			imageSplit.clear();
			cv::Mat image2xBicubic;
			cv::resize(input, image2xBicubic, imageSize, 0, 0, cv::INTER_CUBIC);
			cv::split(image2xBicubic, imageSplit);

			if (!w2xc::convertWithModels(imageY, imageSplit[0], models)) {
				std::cerr << "w2xc::convertWithModels : something error has occured.\nstop." << std::endl;
				return false;
			}

			cv::merge(imageSplit, input);

		} // 2x scaling : end

		if (shrinkRatio != 0.0) {
			cv::Size lastImageSize = input.size();
			lastImageSize.width = lastImageSize.width * shrinkRatio;
			lastImageSize.height = lastImageSize.height * shrinkRatio;
			cv::resize(input, input, lastImageSize, 0, 0, cv::INTER_LINEAR);
		}
	}

	cv::cvtColor(input, output, cv::COLOR_YUV2RGB);
	output.convertTo(output, CV_8U, 255.0);

	return true;
}

int main(int argc, char** argv) {
	time_t start = clock();

	// load image file
	cv::Mat image = cv::imread("../input.png", cv::IMREAD_COLOR);
	image.convertTo(image, CV_32F, 1.0 / 255.0);
	cv::cvtColor(image, image, cv::COLOR_RGB2YUV);

	cv::Mat result;
	if (superres(image, result, 2.0f, false)) {
		cv::imwrite("../result.png", result);

		std::cout << "process successfully done!" << std::endl;
		time_t end = clock();
		std::cout << (double)(end - start) / CLOCKS_PER_SEC << " sec" << std::endl;
	}

	return 0;
}
