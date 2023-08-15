#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <math.h>
#include <tuple>
#include <random>
#include <vector>
#include <tgmath.h>
#include <thread>
#include <mutex>


using namespace std;
using namespace cv;

//std::mutex mtx;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

int ConvertCVGrayImageType(int magicNumber)
{
	magicNumber = (magicNumber >> 8) & 255; //3번째 바이트(픽셀 타입)만 가져오기
	//리틀 엔디안 CPU에서 magicNumber = ((char*)&magicNumber)[1];와 같음
	//빅 엔디안 CPU에서 magicNumber = ((char*)&magicNumber)[2];와 같음

	switch (magicNumber) {
	case 0x08: return CV_8UC1;//unsigned byte, 흑백 채널 단일
	case 0x09: return CV_8SC1;//signed byte, 흑백 채널 단일
	case 0x0B: return CV_16SC1;//short(2 바이트), 흑백 채널 단일
	case 0x0C: return CV_32SC1;//int(4 바이트), 흑백 채널 단일
	case 0x0D: return CV_32FC1;//float(4 바이트), 흑백 채널 단일
	case 0x0E: return CV_64FC1;//double(8 바이트), 흑백 채널 단일
	default: return CV_8UC1;
	}
}

//인자로 받은 vector에 Mnist 훈련 데이터를 파싱해 Matrix으로 저장
void MnistTrainingDataRead(std::string filePath, std::vector<cv::Mat>& vec, int readDataNum)
{
	std::ifstream file(filePath, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		//ifstream::read(str, count)로 count만큼 읽어 str에 저장
		//char은 1바이트, int는 4바이트이므로 int 1개당 char 4개의 정보만큼 가져옴
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		if (readDataNum > number_of_images || readDataNum <= 0)
			readDataNum = number_of_images;

		for (int i = 0; i < readDataNum; ++i)
		{
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, ConvertCVGrayImageType(magic_number));
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					//magicnumber에서 얻은 타입 정보가 unsigned byte 일 경우
					if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
						unsigned char temp = 0;
						file.read((char*)&temp, sizeof(temp));
						tp.at<uchar>(r, c) = (int)temp;
					}
				}
			}
			vec.push_back(tp);
		}
	}
}

void MnistLabelDataRead(std::string filePath, std::vector<uint8_t>& vec, int readDataNum)
{
	std::ifstream file(filePath, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;

		//ifstream::read(str, count)로 count만큼 읽어 str에 저장
		//char은 1바이트, int는 4바이트이므로 int 1개당 char 4개의 정보만큼 가져옴
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		if (readDataNum > number_of_images || readDataNum <= 0)
			readDataNum = number_of_images;

		for (int i = 0; i < readDataNum; ++i)
		{
			//magicnumber에서 얻은 타입 정보가 unsigned byte 일 경우
			if (ConvertCVGrayImageType(magic_number) == CV_8UC1) {
				uint8_t temp = 0;
				file.read((char*)&temp, sizeof(temp));
				vec.push_back(temp);
			}
		}
	}
}

//데이터 확인용 print
void MatPrint(std::vector<cv::Mat>& trainingVec, std::vector<cv::uint8_t>& labelVec)
{
	std::cout << "읽어온 훈련 데이터 수 : " << trainingVec.size() << std::endl;
	std::cout << "읽어온 정답 데이터 수 : " << labelVec.size() << std::endl;

	cv::namedWindow("Window", cv::WINDOW_AUTOSIZE);

	for (int i = 0; i < labelVec.size() && i < trainingVec.size(); i++) {
		imshow("Window", trainingVec[i]);
		std::cout << i << "번째 이미지 정답 : " << (int)labelVec[i] << std::endl;
		//아무 키나 누르면 다음
		/*if (cv::waitKey(0) != -1)
			continue;*/
		waitKey(1);
	}
}

//Mat test(Mat& test1, Mat& test2) {
//	return test1 * test2;
//}

class Softmax {
public:	

	Softmax() {

	}

	
	Eigen::MatrixXd forward(Eigen::MatrixXd& z) {
		//cout << "==============softmax================" << endl;
		double sum = 0;
		Eigen::MatrixXd a;
		a = Eigen::MatrixXd(z.rows(), z.cols());
		/*double max = z.maxCoeff();*/

		//opencv의 normalize를 사용하는건... 모르겠다...

		//a에다가 z 깊은 복사
		for (int i = 0; i < z.rows(); i++) {
			for(int j=0; j < z.cols(); j++)
				a(i, j) = z(i, j);
		}

		a.normalize();
		a /= a.maxCoeff();

		//cout << "normalize 직후 a는" << endl << a << endl;

		//normalization 한 array를 exp해서 a에 담음

		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				sum += exp( a(i, j) ); 
			}
		}

		//cout << " sum은 " <<	 sum << endl;

		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				a(i, j) = exp( a(i, j) ) / sum;
			}
		}

		//normalization


		//cout << "a is " << endl << a << endl;
		//cout << "=====================================" << endl;

		return a;
	}
	
	//Soft-with-loss
	//one-hot encoding vector - a5
	Eigen::MatrixXd backward(Eigen::MatrixXd& a, Eigen::VectorXd& label) {
		double sum = 0;
		Eigen::MatrixXd z = label - a;

		//cout << "z before is " << endl << z << endl;
		//cout << "z after is " << endl << z << endl;
		
		return z;
	}
};



class ReLU {
public:

	ReLU() {

	}

	Eigen::MatrixXd forward( Eigen::MatrixXd& z ) {
		Eigen::MatrixXd a;
		a = Eigen::MatrixXd(z.rows(), z.cols());
		for (int i = 0; i < z.rows(); i++) {
			for (int j = 0; j < z.cols(); j++) {
				if (z(i, j) <= 0) {
					a(i, j) = 0;
				}
				else {
					a(i, j) = z(i, j);
				}
			}
		}

		return a;
	}

	//ReLU
	void backward(Eigen::MatrixXd& z) {
		for (int i = 0; i < z.rows(); i++) {
			for (int j = 0; j < z.cols(); j++) {
				if (z(i, j) <= 0) {
					z(i, j) = 0.0;
				}
				//양수일 때 미분값은 1이므로 그대로 넘어감.
			}
		}
	}

};

//실수 계산을 위한 반올림식
void Round(Eigen::MatrixXd& z, int num) {
	double temp = pow(10, num);
	for (int i = 0; i < z.rows(); i++) {
		for (int j = 0; j < z.cols(); j++) {
			z(i,j) = round(z(i,j) * temp) / temp;
		}
	}
}

void Round_vec(Eigen::VectorXd& z, int num) {
	double temp = pow(10, num);
	for (int i = 0; i < z.rows(); i++) {
		z(i) = round(z(i) * temp) / temp;
	}
}

void remove_too_small(Eigen::MatrixXd& z) {
	for (int i = 0; i < z.rows(); i++) {
		for (int j = 0; j < z.cols(); j++) {
			if (abs( z(i, j) ) < 0.0000001) {
				z(i,j) = 0.0;
			}
		}
	}
}

void remove_too_small_vec(Eigen::VectorXd& z) {
	for (int i = 0; i < z.rows(); i++) {
		if (abs(z(i)) < 0.0000001) {
			z(i) = 0.0;
		}
	}
}

void addition_scalar_to_matrix_no_return(Eigen::MatrixXd& z, double scalar) {
	for (int i = 0; i < z.rows(); i++) {
		for( int j=0; j<z.cols(); j++)
			z(i, j) += scalar;
	}
}

Eigen::MatrixXd addition_scalar_to_matrix(Eigen::MatrixXd& z, double scalar) {
	Eigen::MatrixXd result;
	result = Eigen::MatrixXd(z.rows(), z.cols());
	for (int i = 0; i < z.rows(); i++) {
		for (int j = 0; j < z.cols(); j++)
			result(i, j) = z(i, j) + scalar;
	}
	return result;
}

class MLP {
public:
	//W,b 선언 및 초기화
	Eigen::MatrixXd W1, W2,W3;
	Eigen::VectorXd b1, b2,b3;
	// backward 용 값 저장 matrix
	Eigen::MatrixXd a1, a2,a3;

	//train용 변수들
	//dW,db 선언 및 초기화
	Eigen::MatrixXd dW1, dW2,dW3;
	Eigen::VectorXd db1, db2,db3;

	//가중치 변화량을 담을 변수들
	Eigen::MatrixXd DdW1, DdW2,DdW3;
	Eigen::VectorXd Ddb1, Ddb2, Ddb3;

	//momentum
	Eigen::MatrixXd mdW1, mdW2, mdW3;
	Eigen::VectorXd mdb1, mdb2, mdb3;

	//velocity
	Eigen::MatrixXd vdW1, vdW2,vdW3;
	Eigen::VectorXd vdb1, vdb2,vdb3;

	//cost 선언
	double cost;

	ReLU relu;
	Softmax softmax;

	MLP() {
		relu = ReLU();
		softmax = Softmax();

		W1 = Eigen::MatrixXd(128, 784);
		W2 = Eigen::MatrixXd(128, 128);
		W3 = Eigen::MatrixXd(10, 128);

		b1 = Eigen::VectorXd(128);
		b2 = Eigen::VectorXd(128);
		b3 = Eigen::VectorXd(10);

		//zero로 초기화하기
		W1.setZero();
		W2.setZero();
		W1.setZero();

		b1.setZero();
		b2.setZero();
		b3.setZero();


		dW1 = Eigen::MatrixXd(128, 784);
		dW2 = Eigen::MatrixXd(128, 128);
		dW3 = Eigen::MatrixXd(10, 128);

		db1 = Eigen::VectorXd(128);
		db2 = Eigen::VectorXd(128);
		db3 = Eigen::VectorXd(10);

		//zero로 초기화하기
		dW1.setZero();
		dW2.setZero();
		dW1.setZero();

		db1.setZero();
		db2.setZero();
		db3.setZero();

		DdW1 = Eigen::MatrixXd(128, 784);
		DdW2 = Eigen::MatrixXd(128, 128);
		DdW3 = Eigen::MatrixXd(10, 128);

		Ddb1 = Eigen::VectorXd(128);
		Ddb2 = Eigen::VectorXd(128);
		Ddb3 = Eigen::VectorXd(10);

		//zero로 초기화하기
		DdW1.setZero();
		DdW2.setZero();
		DdW1.setZero();

		Ddb1.setZero();
		Ddb2.setZero();
		Ddb3.setZero();

		mdW1 = Eigen::MatrixXd(128, 784);
		mdW2 = Eigen::MatrixXd(128, 128);
		mdW3 = Eigen::MatrixXd(10, 128);

		mdb1 = Eigen::VectorXd(128);
		mdb2 = Eigen::VectorXd(128);
		mdb3 = Eigen::VectorXd(10);

		//zero로 초기화하기
		mdW1.setZero();
		mdW2.setZero();
		mdW1.setZero();

		mdb1.setZero();
		mdb2.setZero();
		mdb3.setZero();

		vdW1 = Eigen::MatrixXd(128, 784);
		vdW2 = Eigen::MatrixXd(128, 128);
		vdW3 = Eigen::MatrixXd(10, 128);

		vdb1 = Eigen::VectorXd(128);
		vdb2 = Eigen::VectorXd(128);
		vdb3 = Eigen::VectorXd(10);

		//zero로 초기화하기
		vdW1.setZero();
		vdW2.setZero();
		vdW1.setZero();

		vdb1.setZero();
		vdb2.setZero();
		vdb3.setZero();

		//random값 더해주기
		//난수 준비
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0, 250);

		//W1에 random값들 더해주기
		for (int i = 0; i < W1.rows(); i++) {
			for (int j = 0; j < W1.cols(); j++) {
				W1(i, j) += (double)( dist(gen) - 125 ) / 1000.0;

			}

		}
		//W2,b1에 random값들 더해주기
		for (int i = 0; i < W2.rows(); i++) {
			for (int j = 0; j < W2.cols(); j++) {
				W2(i, j) += (double)( dist(gen) - 125 ) / 1000.0;

			}
		}

		for (int i = 0; i < W3.rows(); i++) {
			for (int j = 0; j < W3.cols(); j++) {
				W3(i, j) += (double)(dist(gen) - 125) / 1000.0;

			}
		}

	
		cost = 0.0;

	}



	//return digit, W1,W2,W3,b1,b2,b3,a3
	std::tuple< int, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, 
		Eigen::VectorXd, Eigen::VectorXd,Eigen::VectorXd, Eigen::MatrixXd> predict(Eigen::VectorXd x) {

		Eigen::MatrixXd z1, z2,z3;
		int digit;

		a1.setZero();
		a2.setZero();
		a3.setZero();

		z1 = W1 * x + b1; //W1 is 128 x 784, x is 784 x 1, z1 is 128*1
		a1 = relu.forward(z1); //a1 is 128*1

		z2 = W2 * a1 + b2; //W2 is 128 x 128, a1 is 128 x 1, z2 is 128*1
		a2 = relu.forward(z2); //a2 is 128*1

		z3 = W3 * a2 + b3; //W2 is 10 x 128, a2 is 128 x 1, z3 is 10*1
		a3 = softmax.forward(z3); //a3 is 10*1
	

		digit = 0;
		double max = a3.maxCoeff();
		for (int i = 0; i < 10; i++) {
			if (a3(i) == max) {
				digit = i;
			}
		}

		//return std::make_tuple(digit, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, a5, max);

		//cout << a3 << endl << "===================================================================" << endl;

		return std::make_tuple(digit, W1, W2,W3, b1, b2,b3, a3);
	}

	void backward(Eigen::VectorXd& x, Eigen::VectorXd& Y) {

		//cout << "===================backward start=====================" << endl;

		Eigen::MatrixXd tmp;
		Eigen::MatrixXd tmp1;
		Eigen::MatrixXd tmp2;

		tmp = softmax.backward(a3, Y); //tmp is 10 x 1


		dW3 += tmp * a2.transpose(); //dW3 : 10 x 128, tmp : 10 x 1, a2 : 128 x 1
		db3 += tmp; //db3 : 10 x 1

		tmp1 = W3.transpose() * tmp; // tmp1 : 128 x 1, W3 : 10 x 128, tmp : 10 x 1

		//dW1, db1 계산
		relu.backward(tmp1);

		dW2 += tmp1 * a1.transpose(); //dW2 : 128 x 128, tmp1 : 128 x 1, a1 : 128 x 1
		db2 += tmp1; //db2 : 128 x 1

		tmp2 = W2.transpose() * tmp1; // tmp2 : 128 x 1, W2 : 128 x 128, tmp1 : 128 x 1

		relu.backward(tmp2);

		dW1 += tmp2 * x.transpose(); //dW1 : 128 x 784, tmp2 : 128 x 1, x : 784 x 1
		db1 += tmp2; //db1 : 128 x 1




		//cout << "===================backward end=====================" << endl;

	}

	//각 학습데이터 당 가중치 양을 초기화 하는 것 - 매 학습마다 실행
	void zero_grad() {

		//zero로 초기화하기
		dW1.setZero();
		dW2.setZero();
		dW3.setZero();
		db1.setZero();
		db2.setZero();
		db3.setZero();


	}

	//Adam Optimizer
	void Adam(int iteration, double beta1, double beta2, double lr, double batch_size) {
		int t = iteration + 1;
		
		//momentum update
		mdW1 = beta1 * mdW1 + (1 - beta1) * DdW1;
		mdW2 = beta1 * mdW2 + (1 - beta1) * DdW2;
		mdW3 = beta1 * mdW3 + (1 - beta1) * DdW3;
		mdb1 = beta1 * mdb1 + (1 - beta1) * Ddb1;
		mdb2 = beta1 * mdb2 + (1 - beta1) * Ddb2;
		mdb3 = beta1 * mdb3 + (1 - beta1) * Ddb3;

		//RMSProp update
		vdW1 = beta2 * vdW1 + (1 - beta2) * DdW1.cwiseProduct(DdW1);
		vdW2 = beta2 * vdW2 + (1 - beta2) * DdW2.cwiseProduct(DdW2);
		vdW3 = beta2 * vdW3 + (1 - beta2) * DdW3.cwiseProduct(DdW3);
		vdb1 = beta2 * vdb1 + (1 - beta2) * Ddb1.cwiseProduct(Ddb1);
		vdb2 = beta2 * vdb2 + (1 - beta2) * Ddb2.cwiseProduct(Ddb2);
		vdb3 = beta2 * vdb3 + (1 - beta2) * Ddb3.cwiseProduct(Ddb3);

		//bias correction
		Eigen::MatrixXd mdW1_hat = mdW1 / (1 - pow(beta1, t));
		Eigen::MatrixXd mdW2_hat = mdW2 / (1 - pow(beta1, t));
		Eigen::MatrixXd mdW3_hat = mdW3 / (1 - pow(beta1, t));
		Eigen::MatrixXd mdb1_hat = mdb1 / (1 - pow(beta1, t));
		Eigen::MatrixXd mdb2_hat = mdb2 / (1 - pow(beta1, t));
		Eigen::MatrixXd mdb3_hat = mdb3 / (1 - pow(beta1, t));

		Eigen::MatrixXd vdW1_hat = vdW1 / (1 - pow(beta2, t));
		Eigen::MatrixXd vdW2_hat = vdW2 / (1 - pow(beta2, t));
		Eigen::MatrixXd vdW3_hat = vdW3 / (1 - pow(beta2, t));
		Eigen::MatrixXd vdb1_hat = vdb1 / (1 - pow(beta2, t));
		Eigen::MatrixXd vdb2_hat = vdb2 / (1 - pow(beta2, t));
		Eigen::MatrixXd vdb3_hat = vdb3 / (1 - pow(beta2, t));

		//update
		W1 = W1 - lr * mdW1_hat.cwiseQuotient( addition_scalar_to_matrix( vdW1_hat, pow(1,-8) ) );
		W2 = W2 - lr * mdW2_hat.cwiseQuotient(addition_scalar_to_matrix(vdW2_hat, pow(1, -8)));
		W3 = W3 - lr * mdW3_hat.cwiseQuotient(addition_scalar_to_matrix(vdW3_hat, pow(1, -8)));
		b1 = b1 - lr * mdb1_hat.cwiseQuotient(addition_scalar_to_matrix(vdb1_hat, pow(1, -8)));
		b2 = b2 - lr * mdb2_hat.cwiseQuotient(addition_scalar_to_matrix(vdb2_hat, pow(1, -8)));
		b3 = b3 - lr * mdb3_hat.cwiseQuotient(addition_scalar_to_matrix(vdb3_hat, pow(1, -8)));
		
	}

	void train(int epoch, int batch_size, Eigen::MatrixXd& X, Eigen::VectorXd& Y, float lr, int N) {

		double m = (double)N;

		for (int epoch_n = 0; epoch_n < epoch; epoch_n++) {

			cost = 0.0;

			//zero_grad
			//zero로 초기화하기
			DdW1.setZero();
			DdW2.setZero();
			DdW3.setZero();
			//DdW4.setZero();
			//DdW5.setZero();
			Ddb1.setZero();
			Ddb2.setZero();
			Ddb3.setZero();
			//Ddb4.setZero();
			//Ddb5.setZero();

			cout << "epoch : " << epoch_n << endl;

			//1000장당 1번 update
			for (int iteration = 0; iteration < N / batch_size; iteration++) {


				//cout << " iteration : " << iteration << endl;


				//1000장 학습
				for (int i = iteration * batch_size; i < iteration * batch_size + batch_size; i++) {

					zero_grad();


					//cout << " iteration is " << i << endl;
					//X의 각 row를 predict
					Eigen::VectorXd x = X.row(i); 

					//정규화와 비슷하게 바꿈.
					x /= 255.0;

					Eigen::VectorXd y;
					y = Eigen::VectorXd(10);
					for (int j = 0; j < 10; j++) {
						if (j == (int)Y(i)) {
							y(j) = 1;
						}
						else {
							y(j) = 0;
						}
					}



					std::tuple< int, Eigen::MatrixXd, Eigen::MatrixXd,Eigen::MatrixXd, 
					Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd> predict = this->predict(x);

					
					//digit
					int digit = std::get<0>(predict);
					Eigen::VectorXd y_hat = std::get<7>(predict);
					double max = y_hat.maxCoeff();


					// y * log(y^) 이지만, 정답을 빼곤 모두 y값이 0이므로 간략화
					for (int i = 0; i < 10; i++)
					{
						cost -= y(i) * log(y_hat(i));
					}



					backward(x, y);

					//가중치들을 계산해서 변화량에 담음
					DdW1 += dW1;
					DdW2 += dW2;
					DdW3 += dW3;
					//DdW4 += dW4;
					//DdW5 += dW5;
					Ddb1 += db1;
					Ddb2 += db2;
					Ddb3 += db3;
					//Ddb4 += db4;
					//Ddb5 += db5;

				}


				//bias 평균내기
				double biases[3] = { Ddb1.mean(), Ddb2.mean(), Ddb3.mean()};

				for (int b = 0; b < Ddb1.rows(); b++) {
					Ddb1(b) = biases[0];
				}
				for (int b = 0; b < Ddb2.rows(); b++) {
					Ddb2(b) = biases[1];
				}
				for (int b = 0; b < Ddb3.rows(); b++) {
					Ddb3(b) = biases[2];
				}

				//update
				Adam(iteration, 0.9, 0.999, lr, (double)batch_size);

				//cout << "cost is " << cost / num_all << endl; 


			}

			double num_all = (epoch_n + 1) * m;
			cout << "the number of trained data is " << num_all << endl;

			cout << "cost is " << cost / m << endl;
			
		}



	}

};

int main() {

    std::cout << "Hello OpenCV" << CV_VERSION << std::endl;

	int DataNum = 60000; 

    //read MNIST iamge into OpenCV Mat vector
    std::vector<cv::Mat> trainingVec;
    std::vector<uchar> labelVec;
    MnistTrainingDataRead("datasets\\MNIST\\train-images-idx3-ubyte", trainingVec, DataNum);
    MnistLabelDataRead("datasets\\MNIST\\train-labels-idx1-ubyte", labelVec, DataNum);

	cout << endl << "MNIST Data Read Done" << endl;

	//데이터 랜덤 배열
	std::vector<int> randomIndex;
	for (int i = 0; i < DataNum; i++) {
		randomIndex.push_back(i);
	}

	random_shuffle( randomIndex.begin(), randomIndex.end() );

	std::vector<cv::Mat> random_trainingVec;
	std::vector<uchar> random_labelVec;

	for (int i = 0; i < DataNum; i++) {
		random_trainingVec.push_back(trainingVec[randomIndex[i]]);
		random_labelVec.push_back(labelVec[randomIndex[i]]);
	}

	//학습데이터는 6만장 - 1장은 28 * 28
	
	Eigen::MatrixXd X;
	Eigen::VectorXd Y;
	X = Eigen::MatrixXd(DataNum, 784); //60000 * 784
	Y = Eigen::VectorXd(DataNum);  //60000 * 1

	//trainingVec의 60000개, 각 1개는 28 * 28
	for (int i = 0; i < DataNum; i++) {
		for (int j = 0; j < 784; j++) {
			X(i, j) = (double)random_trainingVec[i].at<uchar>(j / 28, j % 28);
		}

		Y(i) = (double)random_labelVec[i];
	}

	MLP model = MLP();

	/*std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
		Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd,double> result = model.predict(X.row(44));*/

	////멀티스레드 입문
	//vector<thread> threads;

	//vector<Eigen::MatrixXd> vec_tmp_X;
	//vector<Eigen::VectorXd> vec_tmp_Y;

	//for (int i = 0; i < 6; i++) {
	//	Eigen::MatrixXd x_tmp;
	//	Eigen::VectorXd y_tmp;
	//	x_tmp = Eigen::MatrixXd(10000,784);
	//	y_tmp = Eigen::VectorXd(10000);
	//	for (int j = 0; j < 10000; j++) {
	//		for (int k = 0; k < 784; k++) {
	//			x_tmp(j, k) = X(j,k);
	//		}
	//		y_tmp(j) = Y(i * 10000 + j);
	//	}

	//	vec_tmp_X.push_back(x_tmp);
	//	vec_tmp_Y.push_back(y_tmp);

	//}

	//for (int i = 0; i < 6; i++) {
	//	threads.push_back(thread(&MLP::train, ref(model), ref(vec_tmp_X[i]), ref(vec_tmp_Y[i]), 0.4, 10000));
	//}

	//for (int i = 0; i < 6; i++) {
	//	threads[i].join();
	//}
	int epoch = 1;
	int batch_size = 1;

	model.train(epoch, batch_size, X, Y, 0.001, DataNum);
	/*cost = train(X, Y, model, 0.1, DataNum);
	cost = train(X, Y, model, 0.1, DataNum);*/

	cout << endl << "Training Done" << endl;


	cout << " test1 " << endl;
	cout << "answer is " << Y(255) << endl;
	cout << X.row(255) << endl;
	cout << " predict is " << get<0>(model.predict(X.row(255))) << endl;

	cout << get<7>(model.predict(X.row(255))) << endl; //a5
	cout << "==============" << endl;

	cout << " test2 " << endl;
	cout << "answer is " << Y(312) << endl;
	cout << X.row(312) << endl;
	cout << " predict is " << get<0>(model.predict(X.row(312))) << endl;

	cout << get<7>(model.predict(X.row(312))) << endl; //a3
	cout << "==============" << endl;

	cout << " test " << endl;
	cout << "answer is " << Y(401) << endl;
	cout << X.row(401) << endl;
	cout << " predict is " << get<0>(model.predict(X.row(401))) << endl;

	cout << get<7>(model.predict(X.row(401))) << endl; //a3
	cout << "==============" << endl;

    return 0;
}

