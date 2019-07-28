#include "modelHandler.hpp"
// #include <iostream> in modelHandler.hpp
#include <fstream>
#include <thread>

namespace w2xc {
	
bool Model::filter(const std::vector<cv::Mat>& inputPlanes, std::vector<cv::Mat>& outputPlanes) const {
	outputPlanes.clear();
	outputPlanes.resize(nOutputPlanes);

	int nJob = modelUtility::getInstance().getNumberOfJobs();

	// filter job issuing
	std::vector<std::thread> workerThreads;
	int worksPerThread = nOutputPlanes / nJob;
	for (int idx = 0; idx < nJob; idx++) {
		if (!(idx == (nJob - 1) && worksPerThread * nJob != nOutputPlanes)) {
			workerThreads.push_back(
					std::thread(&Model::filterWorker, this,
							std::ref(inputPlanes), std::ref(weights),
							std::ref(outputPlanes),
							static_cast<unsigned int>(worksPerThread * idx),
							static_cast<unsigned int>(worksPerThread)));
		} else {
			// worksPerThread * nJob != nOutputPlanes
			workerThreads.push_back(
					std::thread(&Model::filterWorker, this,
							std::ref(inputPlanes), std::ref(weights),
							std::ref(outputPlanes),
							static_cast<unsigned int>(worksPerThread * idx),
							static_cast<unsigned int>(nOutputPlanes
									- worksPerThread * idx)));
		}
	}
	// wait for finishing jobs
	for (auto& th : workerThreads) {
		th.join();
	}

	return true;
}

bool Model::loadModelFromJSONObject(picojson::object& jsonObj) {
	int matProgress = 0;
	picojson::array &wOutputPlane = jsonObj["weight"].get<picojson::array>();

	// setting weight matrices
	int32_t opIndex = 0;
	for (auto&& wInputPlaneV : wOutputPlane) {
		picojson::array &wInputPlane = wInputPlaneV.get<picojson::array>();

		int32_t ipIndex = 0;
		for (auto&& weightMatV : wInputPlane) {
			picojson::array &weightMat = weightMatV.get<picojson::array>();
			weights[opIndex][ipIndex] = cv::Mat::zeros(kernelSize, kernelSize, CV_32FC1);

			for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
				auto& weightMatRowV = weightMat.at(writingRow);
				picojson::array &weightMatRow = weightMatRowV.get<picojson::array>();

				for (int index = 0; index < kernelSize; index++) {
					weights[opIndex][ipIndex].at<float>(writingRow, index) = weightMatRow[index].get<double>();
				}
			}

			ipIndex++;
		}

		opIndex++;
	}

	// setting biases
	picojson::array biasesData = jsonObj["bias"].get<picojson::array>();
	for (int index = 0; index < nOutputPlanes; index++) {
		biases[index] = biasesData[index].get<double>();
	}

	return true;
}

bool Model::filterWorker(const std::vector<cv::Mat>& inputPlanes, const std::vector<std::vector<cv::Mat>>& weightMatrices, std::vector<cv::Mat>& outputPlanes, unsigned int beginningIndex, unsigned int nWorks) const {
	cv::ocl::setUseOpenCL(false); // disable OpenCL Support(temporary)

	cv::Size ipSize = inputPlanes[0].size();

	for (int opIndex = beginningIndex; opIndex < (beginningIndex + nWorks);	opIndex++) {
		outputPlanes[opIndex] = cv::Mat::zeros(ipSize, CV_32FC1);

		for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) {
			cv::Mat filterOutput = cv::Mat(ipSize, CV_32FC1);
			cv::filter2D(inputPlanes[ipIndex], filterOutput, -1, weightMatrices[opIndex][ipIndex], cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);

			cv::add(outputPlanes[opIndex], filterOutput, outputPlanes[opIndex]);
		}

		cv::add(outputPlanes[opIndex], biases[opIndex], outputPlanes[opIndex]);
		cv::Mat moreThanZero = cv::Mat(ipSize, CV_32FC1);
		cv::Mat lessThanZero = cv::Mat(ipSize, CV_32FC1);
		cv::max(outputPlanes[opIndex], 0.0, moreThanZero);
		cv::min(outputPlanes[opIndex], 0.0, lessThanZero);
		cv::scaleAdd(lessThanZero, 0.1, moreThanZero, outputPlanes[opIndex]);
	}

	return true;
}

modelUtility* modelUtility::instance = nullptr;

modelUtility& modelUtility::getInstance() {
	if (instance == nullptr) {
		instance = new modelUtility();
	}
	return *instance;
}

bool modelUtility::generateModelFromJSON(const std::string& fileName, std::vector<Model>& models) {
	std::ifstream jsonFile;

	jsonFile.open(fileName);
	if (!jsonFile.is_open()) {
		std::cerr << "Error : couldn't open " << fileName << std::endl;
		return false;
	}

	picojson::value jsonValue;
	jsonFile >> jsonValue;
	std::string errMsg = picojson::get_last_error();
	if (!errMsg.empty()) {
		std::cerr << "Error : PicoJSON Error : " << errMsg << std::endl;
		return false;
	}

	picojson::array& objectArray = jsonValue.get<picojson::array>();
	for (auto&& obj : objectArray) {
		models.emplace_back(obj.get<picojson::object>());
	}

	return true;
}

bool modelUtility::setNumberOfJobs(int setNJob) {
	if (setNJob < 1) return false;
	nJob = setNJob;
	return true;
};

int modelUtility::getNumberOfJobs() {
	return nJob;
}

bool modelUtility::setBlockSize(cv::Size size) {
	if(size.width < 0 || size.height < 0)return false;
	blockSplittingSize = size;
	return true;
}

bool modelUtility::setBlockSizeExp2Square(int exp) {
	if (exp < 0) return false;
	int length = std::pow(2, exp);
	blockSplittingSize = cv::Size(length, length);
	return true;
}

cv::Size modelUtility::getBlockSize(){
	return blockSplittingSize;
}

}
