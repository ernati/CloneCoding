#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#pragma once
#include <stdio.h>

#include "opencv2/opencv.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>
#include <tuple>
#include <random>
#include <vector>
#include <tgmath.h>

using namespace std;

//zero padding
Eigen::MatrixXd padding(Eigen::MatrixXd& mat, int padding) {
	Eigen::MatrixXd result = Eigen::MatrixXd(mat.rows() + 2, mat.cols() + 2);
	for (int i = 0; i < result.rows(); i++) {
		for (int j = 0; j < result.cols(); j++) {
			if (i == 0 || i == result.rows() - 1 || j == 0 || j == result.cols() - 1) {
				result(i, j) = 0;
			}
			else {
				result(i, j) = mat(i - 1, j - 1);
			}
		}
	}
	return result;
}

//convolution
//출력 matrix의 크기는 (n+2p-f)/s + 1
Eigen::MatrixXd convolution(Eigen::MatrixXd& mat, int padsize, int stride, int kernelsize, Eigen::MatrixXd& kernel) {
	//make kernel
	Eigen::MatrixXd result = Eigen::MatrixXd((mat.rows() + 2 * padsize - kernelsize) / stride + 1, (mat.rows() + 2 * padsize - kernelsize) / stride + 1);
	result.setZero();

	Eigen::MatrixXd temp;

	//padding
	temp = padding(mat, padsize);

	for (int i = kernelsize / 2; i < temp.rows() - 1; i += stride) {
		for (int j = kernelsize / 2; j < temp.cols() - 1; j += stride) {
			result((i - 1) / stride, (j - 1) / stride) = (temp.block(i - 1, j - 1, kernelsize, kernelsize).cwiseProduct(kernel)).sum();
		}
	}

	return result;
}

//tensor에 filter를 적용했을 때의 함수
Eigen::MatrixXd convolution_filter(vector<Eigen::MatrixXd>& tensor, vector<Eigen::MatrixXd>& filter, int padsize, int stride) {
	Eigen::MatrixXd result;
	int channel = filter.size();

	//filter의 kernel 별로 convolution을 한 후 결과들을 모두 더함.
	for (int i = 0; i < channel; i++) {
		if (i == 0) { result = convolution(tensor[i], padsize, stride, filter[i].rows(), filter[i]); }
		else { result += convolution(tensor[i], padsize, stride, filter[i].rows(), filter[i]); }
	}

	return result;
}

void convolution_channel(std::vector<Eigen::MatrixXd>& result_tensor, std::vector<Eigen::MatrixXd>& tensor, int kernelsize, int input_channel, int output_channel, int padsize, int stride) {
	result_tensor.clear();
	vector< vector<Eigen::MatrixXd> > vector_filter;

	//1. output_channel만큼의 filter를 만듬
	for (int i = 0; i < output_channel; i++) {
		//1-1. input channel 크기의 kernel을 가진 filter를 만듬
		vector<Eigen::MatrixXd> filter;
		for (int j = 0; j < input_channel; j++) {
			Eigen::MatrixXd kernel = Eigen::MatrixXd(kernelsize, kernelsize);
			kernel.setOnes();
			filter.push_back(kernel);
		}

		//1-2. vector filter에 filter를 넣음
		vector_filter.push_back(filter);
	}


	//2. filter 별로 convolution을 진행한 후 합쳐서 result_tensor에 집어 넣기
	for (int i = 0; i < output_channel; i++) {
		result_tensor.push_back(convolution_filter(tensor, vector_filter[i], padsize, stride));
	}


}

class BasicBlock {
public:

};

int main(void) {

}