#include "convertRoutine.hpp"

namespace w2xc {

// converting process inside program
static bool convertWithModelsBasic(const cv::Mat& inputPlane, cv::Mat& outputPlane,	const std::vector<Model>& models);
static bool convertWithModelsBlockSplit(const cv::Mat& inputPlane, cv::Mat& outputPlane, const std::vector<Model>& models);

bool convertWithModels(const cv::Mat& inputPlane, cv::Mat& outputPlane, const std::vector<Model>& models, bool blockSplitting) {
	cv::Size blockSize = modelUtility::getInstance().getBlockSize();
	bool requireSplitting = (inputPlane.size().width * inputPlane.size().height) > blockSize.width * blockSize.height * 3 / 2;

	if (blockSplitting && requireSplitting) {
		return convertWithModelsBlockSplit(inputPlane, outputPlane, models);
	} else {
		//insert padding to inputPlane
		cv::Mat tempMat;
		int nModel = models.size();
		cv::Size outputSize = inputPlane.size();
		cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel, cv::BORDER_REPLICATE);

		bool ret = convertWithModelsBasic(tempMat, outputPlane, models);

		tempMat = outputPlane(cv::Range(nModel, outputSize.height + nModel), cv::Range(nModel, outputSize.width + nModel));
		assert(tempMat.size().width == outputSize.width && tempMat.size().height == outputSize.height);

		tempMat.copyTo(outputPlane);

		return ret;
	}

}

static bool convertWithModelsBasic(const cv::Mat& inputPlane, cv::Mat& outputPlane, const std::vector<Model>& models) {
	// padding is require before calling this function

	std::vector<cv::Mat> inputPlanes = { inputPlane };
	std::vector<cv::Mat> outputPlanes;

	for (int index = 0; index < models.size(); index++) {
		std::cout << "Iteration #" << (index + 1) << "..." << std::endl;
		if (!models[index].filter(inputPlanes, outputPlanes)) {
			std::exit(-1);
		}
		inputPlanes = outputPlanes;
	}

	outputPlanes[0].copyTo(outputPlane);

	return true;
}

static bool convertWithModelsBlockSplit(const cv::Mat& inputPlane, cv::Mat& outputPlane, const std::vector<Model>& models) {
	// padding is not required before calling this function

	// initialize local variables
	cv::Size blockSize = modelUtility::getInstance().getBlockSize();
	unsigned int nModel = models.size();

	//insert padding to inputPlane
	cv::Mat tempMat;
	cv::Size outputSize = inputPlane.size();
	cv::copyMakeBorder(inputPlane, tempMat, nModel, nModel, nModel, nModel, cv::BORDER_REPLICATE);

	// calcurate split rows/cols
	int32_t splitColumns = std::ceil(outputSize.width / (float)(blockSize.width - 2 * nModel));
	int32_t splitRows = std::ceil(outputSize.height / (float)(blockSize.height - 2 * nModel));

	// start to convert
	cv::Mat processRow;
	cv::Mat processBlock;
	cv::Mat processBlockOutput;
	cv::Mat writeMatTo;
	cv::Mat writeMatFrom;
	outputPlane = cv::Mat::zeros(outputSize, CV_32FC1);
	for (unsigned int r = 0; r < splitRows; r++) {
		if (r == splitRows - 1) {
			processRow = tempMat.rowRange(r * (blockSize.height - 2 * nModel), tempMat.size().height);
		} else {
			processRow = tempMat.rowRange(r * (blockSize.height - 2 * nModel), r * (blockSize.height - 2 * nModel) + blockSize.height);
		}
		for (unsigned int c = 0; c < splitColumns; c++) {
			if (c == splitColumns - 1) {
				processBlock = processRow.colRange(
						c * (blockSize.width - 2 * nModel),
						tempMat.size().width);
			} else {
				processBlock = processRow.colRange(
						c * (blockSize.width - 2 * nModel),
						c * (blockSize.width - 2 * nModel) + blockSize.width);
			}

			std::cout << "start process block (" << c << "," << r << ") ..." << std::endl;
			if (!convertWithModelsBasic(processBlock, processBlockOutput, models)) {
				std::cerr << "w2xc::convertWithModelsBasic()\n"
						"in w2xc::convertWithModelsBlockSplit() : \n"
						"something error has occured. stop." << std::endl;
				return false;
			}

			writeMatFrom = processBlockOutput(
					cv::Range(nModel, processBlockOutput.size().height - nModel),
					cv::Range(nModel, processBlockOutput.size().width - nModel));
			writeMatTo = outputPlane(
					cv::Range(r * (blockSize.height - 2 * nModel),
							r * (blockSize.height - 2 * nModel) + processBlockOutput.size().height - 2 * nModel),
					cv::Range(c * (blockSize.height - 2 * nModel),
							c * (blockSize.height - 2 * nModel) + processBlockOutput.size().width - 2 * nModel));
			assert(writeMatTo.size().height == writeMatFrom.size().height && writeMatTo.size().width == writeMatFrom.size().width);
			writeMatFrom.copyTo(writeMatTo);
		}
	}

	return true;
}

}

