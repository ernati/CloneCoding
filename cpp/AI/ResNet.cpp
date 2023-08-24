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

class ReLU_tensor_3D {
public:

	ReLU_tensor_3D() {

	}

	Eigen::Tensor<double, 3>* forward(Eigen::Tensor<double, 3>* tensor) {
		Eigen::Tensor<double, 3>* result = new Eigen::Tensor<double, 3>(tensor->dimension(0), tensor->dimension(1), tensor->dimension(2));

		for (int i = 0; i < tensor->dimension(0); i++) {
			for (int j = 0; j < tensor->dimension(1); j++) {
				for (int k = 0; k < tensor->dimension(2); k++) {
					if ((*tensor)(i, j, k) <= 0) {
						(*result)(i, j, k) = 0;
					}
					else {
						(*result)(i, j, k) = (*tensor)(i, j, k); // 그대로 넘기므로 생략
					}
				}
			}
		}

		return result;
	}
};

class ReLU_tensor_4D {
public:
	ReLU_tensor_4D() {

	}

	Eigen::Tensor<double, 4>* forward(Eigen::Tensor<double, 4>* tensor) {
		Eigen::Tensor<double, 4>* result = new Eigen::Tensor<double, 4>(tensor->dimension(0), tensor->dimension(1), tensor->dimension(2), tensor->dimension(3));

		for (int i = 0; i < tensor->dimension(0); i++) {
			for (int j = 0; j < tensor->dimension(1); j++) {
				for (int k = 0; k < tensor->dimension(2); k++) {
					for (int l = 0; l < tensor->dimension(3); l++) {
						if ((*tensor)(i, j, k, l) <= 0) {
							(*result)(i, j, k, l) = 0;
						}
						else {
							(*result)(i, j, k, l) = (*tensor)(i, j, k, l); // 그대로 넘기므로 생략
						}
					}
				}
			}
		}

		return result;
	}
};

void print_tensor_3d(Eigen::Tensor<double, 3 >* input) {
	Eigen::Tensor< double, 3 >::Dimensions dims = input->dimensions();
	for (int i = 0; i < dims[0]; i++) {
		cout << "( " << i << " )" << endl;
		for (int j = 0; j < dims[1]; j++) {
			for (int k = 0; k < dims[2]; k++) {
				cout << (*input)(i, j, k) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void print_tensor_4d(Eigen::Tensor<double, 4 >* input) {
	Eigen::Tensor< double, 4 >::Dimensions dims = input->dimensions();
	for (int i = 0; i < dims[0]; i++) {
		for (int j = 0; j < dims[1]; j++) {
			cout << "( " << i << ", " << j << " )" << endl;
			for (int k = 0; k < dims[2]; k++) { // row
				for (int l = 0; l < dims[3]; l++) { //col
					cout << (*input)(i, j, k, l) << " ";
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
}

Eigen::Tensor<double, 4>* copy_4D_tensor(Eigen::Tensor<double, 4 >* input) {
	Eigen::Tensor< double, 4 >::Dimensions dims = input->dimensions();
	Eigen::Tensor<double, 4 >* result = new Eigen::Tensor<double, 4>(dims[0], dims[1], dims[2], dims[3]);
	for (int i = 0; i < dims[0]; i++) {
		for (int j = 0; j < dims[1]; j++) {
			for (int k = 0; k < dims[2]; k++) {
				for (int l = 0; l < dims[3]; l++) {
					(*result)(i, j, k, l) = (*input)(i, j, k, l);
				}
			}
		}
	}
	return result;
}

Eigen::Tensor<double, 3 >* copy_3D_tensor(Eigen::Tensor<double, 3 >* input) {
	Eigen::Tensor< double, 3 >::Dimensions dims = input->dimensions();
	Eigen::Tensor<double, 3 >* result = new Eigen::Tensor<double, 3>(dims[0], dims[1], dims[2]);
	for (int i = 0; i < dims[0]; i++) {
		for (int j = 0; j < dims[1]; j++) {
			for (int k = 0; k < dims[2]; k++) {
				(*result)(i, j, k) = (*input)(i, j, k);
			}
		}
	}
	return result;
}

Eigen::Tensor<double, 3> matmul3D(Eigen::Tensor<double, 3>& a2, Eigen::Tensor<double, 3>& a1) {
	//1. result 텐서 선언
	Eigen::Tensor<double, 3> result(a2.dimension(0), a2.dimension(1), a1.dimension(2));

	for (int j = 0; j < a2.dimension(0); j++) {
		//2-1. a2의 tmp 행렬 초기화
		Eigen::MatrixXd tmp_a2 = Eigen::MatrixXd(a2.dimension(2), a2.dimension(3));
		for (int k = 0; k < a2.dimension(1); k++) {
			for (int l = 0; l < a2.dimension(2); l++) {
				tmp_a2(k, l) = a2(j, k, l);
			}
		}

		cout << "tmp_a2's row is " << tmp_a2.rows() << endl;
		cout << "tmp_a2's col is " << tmp_a2.cols() << endl;

		//2-2. a1의 tmp 행렬 초기화
		Eigen::MatrixXd tmp_a1 = Eigen::MatrixXd(a1.dimension(2), a1.dimension(3));
		for (int k = 0; k < a1.dimension(1); k++) {
			for (int l = 0; l < a1.dimension(2); l++) {
				tmp_a1(k, l) = a1(j, k, l);
			}
		}

		cout << "tmp_a1's row is " << tmp_a1.rows() << endl;
		cout << "tmp_a1's col is " << tmp_a1.cols() << endl;

		//2-3. tmp_a2 와 tmp a3 행렬곱
		Eigen::MatrixXd tmp_result = tmp_a2 * tmp_a1;

		//2-4. tmp_result를 result에 저장
		for (int k = 0; k < tmp_result.rows(); k++) {
			for (int l = 0; l < tmp_result.cols(); l++) {
				result(j, k, l) = tmp_result(k, l);
			}
		}
	}

	return result;
}

Eigen::Tensor<double, 4> matmul4D(Eigen::Tensor<double, 4> a2, Eigen::Tensor<double, 4> a1) {
	//1. result 텐서 선언
	Eigen::Tensor<double, 4> result(a2.dimension(0), a2.dimension(1), a2.dimension(2), a1.dimension(3));

	//2. 행렬곱
	for (int i = 0; i < a2.dimension(0); i++) {
		for (int j = 0; j < a2.dimension(1); j++) {
			//2-1. a2의 tmp 행렬 초기화
			Eigen::MatrixXd tmp_a2 = Eigen::MatrixXd(a2.dimension(2), a2.dimension(3));
			for (int k = 0; k < a2.dimension(2); k++) {
				for (int l = 0; l < a2.dimension(3); l++) {
					tmp_a2(k, l) = a2(i, j, k, l);
				}
			}

			cout << "tmp_a2's row is " << tmp_a2.rows() << endl;
			cout << "tmp_a2's col is " << tmp_a2.cols() << endl;
			cout << tmp_a2 << endl;
			cout << "========================================" << endl;

			//2-2. a1의 tmp 행렬 초기화
			Eigen::MatrixXd tmp_a1 = Eigen::MatrixXd(a1.dimension(2), a1.dimension(3));
			for (int k = 0; k < a1.dimension(2); k++) {
				for (int l = 0; l < a1.dimension(3); l++) {
					tmp_a1(k, l) = a1(i, j, k, l);
				}
			}

			cout << "tmp_a1's row is " << tmp_a1.rows() << endl;
			cout << "tmp_a1's col is " << tmp_a1.cols() << endl;
			cout << tmp_a1 << endl;
			cout << "========================================" << endl;

			//2-3. tmp_a2 와 tmp a3 행렬곱
			Eigen::MatrixXd tmp_result = tmp_a2 * tmp_a1;

			cout << "tmp_result's row is " << tmp_result.rows() << endl;
			cout << "tmp_result's col is " << tmp_result.cols() << endl;
			cout << tmp_result << endl;
			cout << "========================================" << endl;


			//2-4. tmp_result를 result에 저장
			for (int k = 0; k < tmp_result.rows(); k++) {
				for (int l = 0; l < tmp_result.cols(); l++) {
					result(i, j, k, l) = tmp_result(k, l);
				}
			}
		}
	}

	return result;

}

//4차원 텐서에서 3차원 텐서 분리
Eigen::Tensor<double, 3>* split4Dto3D(Eigen::Tensor<double, 4>* a, int idx) {

	Eigen::Tensor<double, 3>* result = new Eigen::Tensor<double, 3>(a->dimension(1), a->dimension(2), a->dimension(3));
	result->setZero();
	for (int j = 0; j < a->dimension(1); j++) {
		for (int k = 0; k < a->dimension(2); k++) {
			for (int l = 0; l < a->dimension(3); l++) {
				(*result)(j, k, l) = (*a)(idx, j, k, l);
			}
		}
	}
	return result;
}

//3차원 텐서에서 2차원 텐서 분리
Eigen::MatrixXd split3Dto2D(Eigen::Tensor<double, 3>* a, int idx) {
	Eigen::MatrixXd result(a->dimension(1), a->dimension(2));
	result.setZero();
	for (int j = 0; j < a->dimension(1); j++) {
		for (int k = 0; k < a->dimension(2); k++) {
			result(j, k) = (*a)(idx, j, k);
		}
	}
	return result;
}

//2차원 텐서를 2차원 텐서에다가 넣기
void put2Dto3D(Eigen::MatrixXd a, Eigen::Tensor<double, 3>* result, int idx) {

	for (int i = 0; i < a.rows(); i++) {
		for (int j = 0; j < a.cols(); j++) {
			(*result)(idx, i, j) = a(i, j);
		}
	}
}

//3차원 텐서를 4차원 텐서에다가 넣기
void put3Dto4D(Eigen::Tensor<double, 3> a, Eigen::Tensor<double, 4>* result, int idx) {
	for (int i = 0; i < a.dimension(0); i++) {
		for (int j = 0; j < a.dimension(1); j++) {
			for (int k = 0; k < a.dimension(2); k++) {
				(*result)(idx, i, j, k) = a(i, j, k);
			}
		}
	}
}

////======================================ResNet======================================
//zero padding
Eigen::MatrixXd padding(Eigen::MatrixXd& mat, int padding) {
	Eigen::MatrixXd result = Eigen::MatrixXd(mat.rows() + 2 * padding, mat.cols() + 2 * padding);
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
Eigen::MatrixXd convolution(Eigen::MatrixXd mat, int padsize, int stride, int kernelsize, Eigen::MatrixXd kernel) {
	//make kernel
	Eigen::MatrixXd result = Eigen::MatrixXd((mat.rows() + 2 * padsize - kernelsize) / stride + 1, (mat.rows() + 2 * padsize - kernelsize) / stride + 1);
	result.setZero();

	Eigen::MatrixXd temp;

	if (padsize == 0) { temp = mat; }

	//padding
	else { temp = padding(mat, padsize); }

	if (padsize == 0) {
		for (int i = 0; i < temp.rows(); i += stride) {
			for (int j = 0; j < temp.cols(); j += stride) {
				result(i / stride, j / stride) = (temp.block(i, j, kernelsize, kernelsize).cwiseProduct(kernel)).sum();
			}
		}
	}
	else {
		for (int i = padsize; i < temp.rows() - 1; i += stride) {
			for (int j = padsize; j < temp.cols() - 1; j += stride) {
				result((i - padsize) / stride, (j - padsize) / stride) = (temp.block(i, j, kernelsize, kernelsize).cwiseProduct(kernel)).sum();
			}
		}
	}


	//cout << "=================================" << endl;
	//cout << "input temp" << endl;
	//cout << temp << endl;
	//cout << "=================================" << endl;

	//cout << "=================================" << endl;
	//cout << "convolution result" << endl;
	//cout << result << endl;
	//cout << "=================================" << endl;

	return result;
}

//tensor에 filter를 적용했을 때의 함수
Eigen::MatrixXd convolution_filter(Eigen::Tensor<double, 3>* tensor, Eigen::Tensor<double, 3>* filter, int padsize, int stride, int image_row, int image_col) {
	Eigen::MatrixXd result;
	int num_of_kernel = filter->dimension(0);

	//filter의 kernel 별로 convolution을 한 후 결과들을 모두 더함.
	for (int i = 0; i < num_of_kernel; i++) {

		if (i == 0) { result = convolution(split3Dto2D(tensor, i), padsize, stride, split3Dto2D(filter, i).rows(), split3Dto2D(filter, i)); }
		else { result = result + convolution(split3Dto2D(tensor, i), padsize, stride, split3Dto2D(filter, i).rows(), split3Dto2D(filter, i)); }
	}

	return result;
}

//input_channel개 만큼 matrix를 가지고 있는 tensor를 입력받아서, output_channel개 만큼 matrix를 가지고 있는 tensor를 출력하는 함수
Eigen::Tensor<double, 3>* convolution_channel(Eigen::Tensor<double, 3>* tensor, Eigen::Tensor<double, 4>* filters, int kernelsize, int input_channel, int output_channel, int padsize, int stride, int image_row, int image_col) {
	//result_tensor : output_channel개 만큼 matrix를 가지고 있는 tensor

	Eigen::Tensor<double, 3>* result_tensor = new Eigen::Tensor<double, 3>(output_channel, image_row, image_col);
	result_tensor->setZero();

	Eigen::Tensor<double, 3>* tmp_filter = new Eigen::Tensor<double, 3>(input_channel, kernelsize, kernelsize);

	//filter 별로 convolution을 진행한 후 합쳐서 result_tensor에 집어 넣기
	for (int i = 0; i < output_channel; i++) {
		tmp_filter = split4Dto3D(filters, i);
		put2Dto3D(convolution_filter(tensor, tmp_filter, padsize, stride, image_row, image_col), result_tensor, i);
	}

	return result_tensor;
}

class BasicBlock {
public:
	ReLU_tensor_3D relu_3d;
	ReLU_tensor_4D relu_4d;
	int kernelsize;
	int expansion;
	int input_channel;
	int output_channel;

	int image_size_row;
	int image_size_col;

	//parameters
	Eigen::Tensor<double, 4>* filters_convolution_1; //filter 안의 kernel은 inputchannel 개, filter는 총 outputchannel 개
	Eigen::Tensor<double, 4>* filters_convolution_2; //filter 안의 kernel은 outputchannel 개, filter는 총 outputchannel 개
	Eigen::Tensor<double, 4>* filters_convolution_3; //filter 안의 kernel은 outputchannel 개, filter는 총 outputchannel 개

	//forward과정에 쓰일 tensor
	Eigen::Tensor<double, 3>* output_tensor1;
	Eigen::Tensor<double, 3>* output_tensor2;
	Eigen::Tensor<double, 3>* output_tensor3;
	Eigen::Tensor<double, 3>* output_tensor4;

	BasicBlock() {

	}

	//filter 초기화
	BasicBlock(int input_channel, int output_channel, int kernelsize, int image_size_row, int image_size_col) {
		//vector_filter_convolution_1 initialize
		filters_convolution_1 = new Eigen::Tensor<double, 4>(output_channel, input_channel, kernelsize, kernelsize);
		filters_convolution_2 = new Eigen::Tensor<double, 4>(output_channel, output_channel, kernelsize, kernelsize);
		filters_convolution_3 = new Eigen::Tensor<double, 4>(output_channel, output_channel, 1, 1);

		//output_tensor initialize
		output_tensor1 = new Eigen::Tensor<double, 3>(output_channel, image_size_row, image_size_col);
		output_tensor2 = new Eigen::Tensor<double, 3>(output_channel, image_size_row, image_size_col);
		output_tensor3 = new Eigen::Tensor<double, 3>(output_channel, image_size_row, image_size_col);
		output_tensor4 = new Eigen::Tensor<double, 3>(output_channel, image_size_row, image_size_col);

		expansion = 1;
		relu_3d = ReLU_tensor_3D();
		relu_4d = ReLU_tensor_4D();

		//variable initialize
		this->kernelsize = kernelsize;
		this->input_channel = input_channel;
		this->output_channel = output_channel;
		this->image_size_col = image_size_col;
		this->image_size_row = image_size_row;

		//filters initialize
		filters_convolution_1->setConstant(1.0f);
		filters_convolution_2->setConstant(1.0f);
		filters_convolution_3->setConstant(1.0f);

		//output_tensor initialize
		output_tensor1->setZero();
		output_tensor2->setZero();
		output_tensor3->setZero();
		output_tensor4->setZero();
	}

	//입력 : 3차원 텐서
	Eigen::Tensor<double, 3>* forward(Eigen::Tensor<double, 3>* x) {

		cout << "===========================================" << endl;
		cout << "forward start" << endl;

		//1. input -> output
		//x is 3 x 8 x 8
		output_tensor1 = convolution_channel(x, filters_convolution_1, this->kernelsize, input_channel, output_channel, 1, 1, image_size_row, image_size_col);

		//output_tensor1 is 16 x 8 x 8

		//2. output -> output
		output_tensor2 = convolution_channel(output_tensor1, filters_convolution_2, this->kernelsize, output_channel, output_channel, 1, 1, image_size_row, image_size_col); // outputchannel x image_row x image_col

		//3. output -> output ( 1x1 kernel )
		output_tensor3 = convolution_channel(output_tensor2, filters_convolution_3, 1, output_channel, output_channel, 0, 1, image_size_row, image_size_col);	// outputchannel x image_row x image_col

		//4. relu
		output_tensor4 = relu_3d.forward(output_tensor3);

		return output_tensor4;
	}

	void delete_data() {
		delete filters_convolution_1;
		delete filters_convolution_2;
		delete filters_convolution_3;

		delete output_tensor1;
		delete output_tensor2;
		delete output_tensor3;
		delete output_tensor4;
	}
};

int main(void) {
	BasicBlock block = BasicBlock(3, 16, 3, 8, 8);

	input_tensor->setConstant(2.0f);

	Eigen::Tensor<double, 3>* result_tensor = block.forward(input_tensor);

	cout << "============================================" << endl;
	cout << "intput_tensor is " << endl;
	print_tensor_3d(input_tensor);
	cout << "============================================" << endl;

	cout << "============================================" << endl;
	cout << "forward result is " << endl;
	print_tensor_3d(result_tensor);
	cout << "============================================" << endl;
}