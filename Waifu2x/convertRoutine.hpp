/*
 * convertRoutine.hpp
 *   (ここにファイルの簡易説明を記入)
 *
 *  Created on: 2015/05/31
 *      Author: wlamigo
 * 
 *   (ここにファイルの説明を記入)
 */

#ifndef CONVERTROUTINE_HPP_
#define CONVERTROUTINE_HPP_

#include "modelHandler.hpp"
#include <memory>
//#include "opencv2/opencv.hpp"
//#include "opencv2/core/ocl.hpp" in modelHandler.hpp
#include <vector>

namespace w2xc {

/**
 * convert inputPlane to outputPlane by convoluting with models.
 */
bool convertWithModels(const cv::Mat& inputPlanes, cv::Mat &outputPlanes, const std::vector<Model>& models, bool blockSplitting = true);

}



#endif /* CONVERTROUTINE_HPP_ */
