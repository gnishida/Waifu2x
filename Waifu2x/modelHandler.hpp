#ifndef MODEL_HANDLER_HPP_
#define MODEL_HANDLER_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "json.h"
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
	bool loadModelFromJSONObject(const nlohmann::json& jsonObj);

	// thread worker function
	bool filterWorker(const std::vector<cv::Mat>& inputPlanes, const std::vector<std::vector<cv::Mat>>& weightMatrices, std::vector<cv::Mat>& outputPlanes, unsigned int beginningIndex, unsigned int nWorks) const;

public:
	Model(const nlohmann::json& jsonObj) {
		nInputPlanes = jsonObj["nInputPlane"].get<double>();
		nOutputPlanes = jsonObj["nOutputPlane"].get<double>();
		kernelSize = jsonObj["kW"].get<double>();
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

	static bool generateModelFromJSON(const std::string& fileName, std::vector<Model>& models);
};

}

#endif /* MODEL_HANDLER_HPP_ */
