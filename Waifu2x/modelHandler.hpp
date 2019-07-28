#ifndef MODEL_HANDLER_HPP_
#define MODEL_HANDLER_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "picojson.h"
#include <iostream>
#include <memory>
#include <cstdint>
#include <cstdlib>

namespace w2xc {

class Model {

private:
	int nInputPlanes;
	int nOutputPlanes;
	std::vector<std::vector<cv::Mat>> weights;
	std::vector<float> biases;
	int kernelSize;

	Model() {}

	// class inside operation function
	bool loadModelFromJSONObject(picojson::object& jsonObj);

	// thread worker function
	bool filterWorker(const std::vector<cv::Mat>& inputPlanes, const std::vector<std::vector<cv::Mat>>& weightMatrices, std::vector<cv::Mat>& outputPlanes, unsigned int beginningIndex, unsigned int nWorks) const;

public:
	// ctor and dtor
	Model(picojson::object &jsonObj) {
		nInputPlanes = static_cast<int>(jsonObj["nInputPlane"].get<double>());
		nOutputPlanes = static_cast<int>(jsonObj["nOutputPlane"].get<double>());
		kernelSize = static_cast<int>(jsonObj["kW"].get<double>());
		weights.resize(nOutputPlanes, std::vector<cv::Mat>(nInputPlanes));
		biases = std::vector<float>(nOutputPlanes, 0.0);

		if (!loadModelFromJSONObject(jsonObj)) {
			std::cerr << "Error : Model-Constructor : \n"
							"something error has been occured in loading model from JSON-Object.\n"
							"stop." << std::endl;
			std::exit(-1);
		}
	}
	
	// public operation function
	bool filter(const std::vector<cv::Mat>& inputPlanes, std::vector<cv::Mat>& outputPlanes) const;
};

class modelUtility {

private:
	static modelUtility* instance;
	int nJob;
	cv::Size blockSplittingSize;
	modelUtility() : nJob(4), blockSplittingSize(512,512) {}

public:
	static bool generateModelFromJSON(const std::string& fileName, std::vector<Model>& models);
	static modelUtility& getInstance();
	bool setNumberOfJobs(int setNJob);
	int getNumberOfJobs();
	bool setBlockSize(cv::Size size);
	bool setBlockSizeExp2Square(int exp);
	cv::Size getBlockSize();
};

}

#endif /* MODEL_HANDLER_HPP_ */
