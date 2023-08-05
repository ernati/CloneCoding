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
			for(int j=0; j<z.cols(); j++)
				a(i, j) = z(i, j);
		}

		a.normalize();

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
	
	//one-hot encoding vector - a5
	Eigen::MatrixXd backward(Eigen::MatrixXd& a, Eigen::VectorXd& label) {
		double sum = 0;
		Eigen::MatrixXd z = label - a;
		
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
				if (z(i, j) < 0) {
					a(i, j) = 0;
				}
				else {
					a(i, j) = z(i, j);
				}
			}
		}

		return a;
	}

	void backward(Eigen::MatrixXd& z) {
		for (int i = 0; i < z.rows(); i++) {
			for (int j = 0; j < z.cols(); j++) {
				if (z(i, j) < 0) {
					z(i, j) *= 0;
				}
				else {
					z(i, j) *= 1;
				}
			}
		}
	}

};

class MLP {
public:
	//W,b 선언 및 초기화
	Eigen::MatrixXd W1, W2, W3, W4, W5;
	Eigen::VectorXd b1, b2, b3, b4, b5;
	// backward 용 값 저장 matrix
	Eigen::MatrixXd a1, a2, a3, a4, a5;

	//train용 변수들
	//dW,db 선언 및 초기화
	Eigen::MatrixXd dW1, dW2, dW3, dW4, dW5;
	Eigen::VectorXd db1, db2, db3, db4, db5;

	//cost 선언
	double cost;

	ReLU relu;
	Softmax softmax;

	MLP() {
		relu = ReLU();

		W1 = Eigen::MatrixXd(512, 784);
		W2 = Eigen::MatrixXd(256, 512);
		W3 = Eigen::MatrixXd(128, 256);
		W4 = Eigen::MatrixXd(64, 128);
		W5 = Eigen::MatrixXd(10, 64);

		b1 = Eigen::VectorXd(512);
		b2 = Eigen::VectorXd(256);
		b3 = Eigen::VectorXd(128);
		b4 = Eigen::VectorXd(64);
		b5 = Eigen::VectorXd(10);

		//zero로 초기화하기
		W1.setZero();
		W2.setZero();
		W3.setZero();
		W4.setZero();
		W5.setZero();
		b1.setZero();
		b2.setZero();
		b3.setZero();
		b4.setZero();
		b5.setZero();

		dW1 = Eigen::MatrixXd(512, 784);
		dW2 = Eigen::MatrixXd(256, 512);
		dW3 = Eigen::MatrixXd(128, 256);
		dW4 = Eigen::MatrixXd(64, 128);
		dW5 = Eigen::MatrixXd(10, 64);

		db1 = Eigen::VectorXd(512);
		db2 = Eigen::VectorXd(256);
		db3 = Eigen::VectorXd(128);
		db4 = Eigen::VectorXd(64);
		db5 = Eigen::VectorXd(10);

		//zero로 초기화하기
		dW1.setZero();
		dW2.setZero();
		dW3.setZero();
		dW4.setZero();
		dW5.setZero();
		db1.setZero();
		db2.setZero();
		db3.setZero();
		db4.setZero();
		db5.setZero();

		//random값 더해주기
		//난수 준비
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0, 1000);

		//W1에 random값들 더해주기
		for (int i = 0; i < 784; i++) {
			for (int j = 0; j < 512; j++) {
				W1(i, j) = W1(i, j) + ((double)dist(gen) - 500) / 1000.0;
				dW1(i, j) = dW1(i, j) + ((double)dist(gen) - 500) / 1000.0;
			}
		}
		//W2,b1에 random값들 더해주기
		for (int i = 0; i < 512; i++) {
			for (int j = 0; j < 256; j++) {
				W2(i, j) = W2(i, j) + ((double)dist(gen) - 500) / 1000.0;
				dW2(i, j) = dW2(i, j) + ((double)dist(gen) - 500) / 1000.0;
			}
			b1(i) = b1(i) + ((double)dist(gen) - 500) / 1000.0;
			db1(i) = db1(i) + ((double)dist(gen) - 500) / 1000.0;
		}
		//W3,b2에 random값들 더해주기
		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 128; j++) {
				W3(i, j) = W3(i, j) + ((double)dist(gen) - 500) / 1000.0;
				dW3(i, j) = dW3(i, j) + ((double)dist(gen) - 500) / 1000.0;
			}
			b2(i) = b2(i) + ((double)dist(gen) - 500) / 1000.0;
			db2(i) = db2(i) + ((double)dist(gen) - 500) / 1000.0;
		}
		//W4, b3에 random값들 더해주기
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 64; j++) {
				W4(i, j) = W4(i, j) + ((double)dist(gen) - 500) / 1000.0;
				dW4(i, j) = dW4(i, j) + ((double)dist(gen) - 500) / 1000.0;
			}
			b3(i) = b3(i) + ((double)dist(gen) - 500) / 1000.0;
			db3(i) = db3(i) + ((double)dist(gen) - 500) / 1000.0;
		}
		//W5, b4에 random값들 더해주기
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 10; j++) {
				W5(i, j) = W5(i, j) + ((double)dist(gen) - 500) / 1000.0;
				dW5(i, j) = dW5(i, j) + ((double)dist(gen) - 500) / 1000.0;
			}
			b4(i) = b4(i) + ((double)dist(gen) - 500) / 1000.0;
			db4(i) = db4(i) + ((double)dist(gen) - 500) / 1000.0;
		}

		//b5에 random 값들 더해주기
		for (int i = 0; i < 10; i++) {
			b5(i) = b5(i) + ((double)dist(gen) - 500) / 1000.0;
			db5(i) = db5(i) + ((double)dist(gen) - 500) / 1000.0;
		}

		cost = 0.0;

	}


	//return a6, w1,w2,w3,w4,w5,b1,b2,b3,b4,b5
	std::tuple< int, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
		Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, double> predict(Eigen::VectorXd x) {

		Eigen::MatrixXd z1, z2, z3, z4, z5;
		int digit;
		z1 = W1 * x + b1;
		a1 = relu.forward(z1);
		z2 = W2 * a1 + b2;
		a2 = relu.forward(z2);
		z3 = W3 * a2 + b3;
		a3 = relu.forward(z3);
		z4 = W4 * a3 + b4;
		a4 = relu.forward(z4);
		z5 = W5 * a4 + b5;
		a5 = softmax.forward(z5);

		digit = 0;
		double max = a5(0);
		for (int i = 0; i < 10; i++) {
			if (a5(i) > max) {
				digit = i;
				max = a5(i);
			}
		}

		/*cout << a1.rows() << " " << a1.cols() << endl;
		cout << a2.rows() << " " << a2.cols() << endl;
		cout << a3.rows() << " " << a3.cols() << endl;
		cout << a4.rows() << " " << a4.cols() << endl;
		cout << a5.rows() << " " << a5.cols() << endl;
		cout << a5 << endl;
		cout << y_hat << endl;*/

		/*cout << "a3 is " << endl << a3 << endl;
		cout << "=========================" << endl;*/
		//cout << "a4 is " << endl << a4 << endl;
		//cout << "=========================" << endl;
		//cout << "a5 is " << endl << a5 << endl;
		//cout << "=========================" << endl;

		/*cout << "=========================" << endl;
		cout << " z4 is " << z4 << endl;
		cout << "=========================" << endl;*/

		return std::make_tuple(digit, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, a5, max);
	}

	void backward(Eigen::VectorXd& x, Eigen::VectorXd& Y) {

		//cout << "===================backward start=====================" << endl;

		Eigen::MatrixXd tmp;

		//cout << "dW5 start" << endl;

		//dW5, db5 계산
		tmp = softmax.backward(a5, Y); //Y is 10x1 one hot encoding vector

		/*cout << "a5 is " << a5 << endl << endl;
		cout << "Y is " << Y << endl << endl;*/

		//cout << "tmp is " << tmp << endl << endl;

		dW5 += tmp * a4.transpose();
		db5 += tmp;
		tmp = W5.transpose() * tmp;

		/*cout << "tmp is " << tmp << endl << endl;

		cout << "dW5 end" << endl << endl;
		cout << "dW4 start" << endl;*/

		//dW4, db4 계산
		relu.backward(tmp);
		dW4 += tmp * a3.transpose();
		db4 += tmp;
		tmp = W4.transpose() * tmp;

		/*cout << "tmp is " << tmp << endl << endl;

		cout << "dW4 end" << endl << endl;
		cout << "dW3 start" << endl;*/

		//dW3, db3 계산
		relu.backward(tmp);

		//cout << "tmp is " << tmp << endl << endl;

		dW3 += tmp * a2.transpose();
		//cout << "dW3 is " << dW3 << endl << endl;

		db3 += tmp;
		tmp = W3.transpose() * tmp;

		//cout << "tmp is " << tmp << endl << endl;

		//cout << "dW3 end" << endl << endl;
		//cout << "dW2 start" << endl;

		//dW2, db2 계산
		relu.backward(tmp);

		/*cout << "tmp's row : " << tmp.rows() << endl << endl;
		cout << "tmp's row : " << tmp.cols() << endl << endl;*/

		dW2 += tmp * a1.transpose();

		/*cout << "dW2's row " << dW2.rows() << endl << endl;
		cout << "dW2's col " << dW2.cols() << endl << endl;*/

		db2 += tmp;

		tmp = W2.transpose() * tmp;

		/*cout << "dW2 end" << endl << endl;
		cout << "dW1 start" << endl;*/

		//dW1, db1 계산
		relu.backward(tmp);
		dW1 += tmp * x.transpose();
		db1 += tmp;

		//cout << "===================backward end=====================" << endl;

	}

	void train(Eigen::MatrixXd& X, Eigen::VectorXd& Y, float lr, int N) {

		double m = (double)N;

		int epoch_size = 1000;

		//10장당 1번 update
		for (int epoch = 0; epoch < N / epoch_size; epoch++) {


			cout << " epoch : " << epoch << endl;


			//한 epoch당 100장씩 train
			for (int i = epoch * epoch_size; i < epoch * epoch_size + epoch_size; i++) {



				//cout << " iteration is " << i << endl;
				//X의 각 row를 predict
				Eigen::VectorXd x = X.row(i);
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
				//std::cout << "Y(i) is " << Y(i) << std::endl;
				//std::cout << "y is " << endl << y << std::endl

				//mtx.lock();

				//digit, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5 
				std::tuple< int, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
					Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, double> predict = this->predict(x);
				//digit
				int digit = std::get<0>(predict);
				double y_hat = std::get<12>(predict);
				Eigen::VectorXd one_vector_b5 = Eigen::VectorXd(10);
				one_vector_b5.setOnes();

				//loss 함수
				cost -= 1 * log(y_hat);



				this->backward(x, y);

				//mtx.unlock();
			}

			//mtx.lock();

			//update
			double num = (epoch + 1) * epoch_size;
			//double num = epoch_size;

			this->cost /= num;
			this->W1 -= lr * dW1 / num;
			this->b1 -= lr * db1 / num;
			this->W2 -= lr * dW2 / num;
			this->b2 -= lr * db2 / num;
			this->W3 -= lr * dW3 / num;
			this->b3 -= lr * db3 / num;
			this->W4 -= lr * dW4 / num;
			this->b4 -= lr * db4 / num;
			this->W5 -= lr * dW5 / num;
			this->b5 -= lr * db5 / num;

			cout << "cost is " << cost << endl;

			//mtx.unlock();

		}

	}

};

////double train(Eigen::MatrixXd& X, Eigen::VectorXd& Y, MLP& model, float lr, int N) {
//	////dW,db 선언 및 초기화
//	//Eigen::MatrixXd dW1, dW2, dW3, dW4, dW5;
//	//Eigen::VectorXd db1, db2, db3, db4, db5;
//	/*dW1 = Eigen::MatrixXd(512, 784);
//	dW2 = Eigen::MatrixXd(256, 512);
//	dW3 = Eigen::MatrixXd(128, 256);
//	dW4 = Eigen::MatrixXd(64, 128);
//	dW5 = Eigen::MatrixXd(10, 64);
//
//	db1 = Eigen::VectorXd(512);
//	db2 = Eigen::VectorXd(256);
//	db3 = Eigen::VectorXd(128);
//	db4 = Eigen::VectorXd(64);
//	db5 = Eigen::VectorXd(10);*/
//
//	////zero로 초기화하기
//	//dW1.setZero();
//	//dW2.setZero();
//	//dW3.setZero();
//	//dW4.setZero();
//	//dW5.setZero();
//	//db1.setZero();
//	//db2.setZero();
//	//db3.setZero();
//	//db4.setZero();
//	//db5.setZero();
//
//	////random값 더해주기
//	//	//난수 준비
//	//std::random_device rd;
//	//std::mt19937 gen(rd());
//	//std::uniform_int_distribution<int> dist(0, 1000);
//
//	////W1에 random값들 더해주기
//	//for (int i = 0; i < 784; i++) {
//	//	for (int j = 0; j < 512; j++) {
//	//		dW1(i, j) = dW1(i, j) + ((double)dist(gen) - 500) / 1000.0;
//	//	}
//	//}
//	////W2,b1에 random값들 더해주기
//	//for (int i = 0; i < 512; i++) {
//	//	for (int j = 0; j < 256; j++) {
//	//		dW2(i, j) = dW2(i, j) + ((double)dist(gen) - 500) / 1000.0;
//	//	}
//	//	db1(i) = db1(i) + ((double)dist(gen) - 500) / 1000.0;
//	//}
//	////W3,b2에 random값들 더해주기
//	//for (int i = 0; i < 256; i++) {
//	//	for (int j = 0; j < 128; j++) {
//	//		dW3(i, j) = dW3(i, j) + ((double)dist(gen) - 500) / 1000.0;
//	//	}
//	//	db2(i) = db2(i) + ((double)dist(gen) - 500) / 1000.0;
//	//}
//	////W4, b3에 random값들 더해주기
//	//for (int i = 0; i < 128; i++) {
//	//	for (int j = 0; j < 64; j++) {
//	//		dW4(i, j) = dW4(i, j) + ((double)dist(gen) - 500) / 1000.0;
//	//	}
//	//	db3(i) = db3(i) + ((double)dist(gen) - 500) / 1000.0;
//	//}
//	////W5, b4에 random값들 더해주기
//	//for (int i = 0; i < 64; i++) {
//	//	for (int j = 0; j < 10; j++) {
//	//		dW5(i, j) = dW5(i, j) + ((double)dist(gen) - 500) / 1000.0;
//	//	}
//	//	db4(i) = db4(i) + ((double)dist(gen) - 500) / 1000.0;
//	//}
//
//	////b5에 random 값들 더해주기
//	//for (int i = 0; i < 10; i++) {
//	//	db5(i) = db5(i) + ((double)dist(gen) - 500) / 1000.0;
//	//}
//
//	//double m = (double)N;
//	//double cost = 0.0;
//
//	//int epoch_size = 1000;
//
//	////10장당 1번 update
//	//for (int epoch = 0; epoch < N / epoch_size; epoch++) {
//
//
//	//	//cout << " epoch : " << epoch << endl;
//	//	
//
//	//	//한 epoch당 100장씩 train
//	//	for (int i = epoch * epoch_size; i < epoch * epoch_size + epoch_size; i++) {
//
//
//
//	//		//cout << " iteration is " << i << endl;
//	//		//X의 각 row를 predict
//	//		Eigen::VectorXd x = X.row(i);
//	//		Eigen::VectorXd y;
//	//		y = Eigen::VectorXd(10);
//	//		for (int j = 0; j < 10; j++) {
//	//			if (j == (int)Y(i)) {
//	//				y(j) = 1;
//	//			}
//	//			else {
//	//				y(j) = 0;
//	//			}
//	//		}
//	//		//std::cout << "Y(i) is " << Y(i) << std::endl;
//	//		//std::cout << "y is " << endl << y << std::endl
//
//	//		mtx.lock();
//
//	//		//digit, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5 
//	//		std::tuple< int, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
//	//			Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, double> predict = model.predict(x);
//	//		//digit
//	//		int digit = std::get<0>(predict);
//	//		double y_hat = std::get<12>(predict);
//	//		Eigen::VectorXd one_vector_b5 = Eigen::VectorXd(10);
//	//		one_vector_b5.setOnes();
//
//	//		//loss 함수
//	//		cost -= 1 * log(y_hat);
//
//	//		
//
//	//		model.backward(dW1, dW2, dW3, dW4, dW5, db1, db2, db3, db4, db5, x, y);
//
//	//		mtx.unlock();
//	//	}
//
//	//	mtx.lock();
//
//	//	//update
//	//	double num = (epoch + 1) * epoch_size;
//	//	//double num = epoch_size;
//
//	//	cost /= num;
//	//	model.W1 -= lr * dW1 / num;
//	//	model.b1 -= lr * db1 / num;
//	//	model.W2 -= lr * dW2 / num;
//	//	model.b2 -= lr * db2 / num;
//	//	model.W3 -= lr * dW3 / num;
//	//	model.b3 -= lr * db3 / num;
//	//	model.W4 -= lr * dW4 / num;
//	//	model.b4 -= lr * db4 / num;
//	//	model.W5 -= lr * dW5 / num;
//	//	model.b5 -= lr * db5 / num;
//
//	//	cout << "cost is " << cost << endl;
//
//	//	mtx.unlock();
//
//	//}
//
//
//
//	//return cost;
//
////}



int main() {

    std::cout << "Hello OpenCV" << CV_VERSION << std::endl;

	int DataNum = 60000;

    //read MNIST iamge into OpenCV Mat vector
    std::vector<cv::Mat> trainingVec;
    std::vector<uchar> labelVec;
    MnistTrainingDataRead("datasets\\MNIST\\train-images-idx3-ubyte", trainingVec, DataNum);
    MnistLabelDataRead("datasets\\MNIST\\train-labels-idx1-ubyte", labelVec, DataNum);

	cout << endl << "MNIST Data Read Done" << endl;

    //MatPrint(trainingVec, labelVec);

	/*cout << trainingVec[50] << endl;
	cout << (int)trainingVec[50].at<uchar>(0, 0) << endl;*/

	//cout << (int)labelVec[99] << endl;
	//cout << labelVec.size() << endl;



	//Mat test1, test2, test3;
	//Vec<float, 512> test_vec;

	//test1 = Mat::ones(784, 512, CV_32FC1);
	//test2 = Mat::ones(512, 10, CV_32FC1);

	//Mat test4 = Mat::ones(4, 3, CV_32FC1);

	//test_vec << 1, 1, 1 ; //512 x 1

	//test3 = test(test1, test2);

	////cout << test3.at<float>(300,300) << endl;

	//cout << (test1 * test_vec) << endl;

	//학습데이터는 6만장 - 1장은 28 * 28
	
	Eigen::MatrixXd X;
	Eigen::VectorXd Y;
	X = Eigen::MatrixXd(DataNum, 784); //60000 * 784
	Y = Eigen::VectorXd(DataNum);  //60000 * 1

	//trainingVec의 60000개, 각 1개는 28 * 28
	for (int i = 0; i < DataNum; i++) {
		for (int j = 0; j < 784; j++) {
			X(i, j) = (double)(int)trainingVec[i].at<uchar>(j / 28, j % 28);  // 가장 중요한 부분!
		}

		Y(i) = (double)(int)labelVec[i];
	}

	MLP model = MLP();

	cout << endl << "MNIST to Eigen Done" << endl;

	/*std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
		Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd,double> result = model.predict(X.row(44));*/

	////멀티스레드 입문
	//vector<thread> threads;

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

	//	threads.push_back(thread(&MLP::train, model, x_tmp, y_tmp, 0.4, 10000));
	//}

	//for (int i = 0; i < 6; i++) {
	//	threads[i].join();
	//}

	model.train(X, Y, 0.4, DataNum);
	/*cost = train(X, Y, model, 0.1, DataNum);
	cost = train(X, Y, model, 0.1, DataNum);*/

	cout << endl << "Training Done" << endl;


	cout << " test1 " << endl;
	cout << "answer is " << Y(255) << endl;
	cout << X.row(255) << endl;
	cout << " predict is " << get<0>(model.predict(X.row(255))) << endl;

	cout << get<11>(model.predict(X.row(255))) << endl; //a5
	cout << get<12>(model.predict(X.row(255))) << endl; //y_hat
	cout << "==============" << endl;

	cout << " test2 " << endl;
	cout << "answer is " << Y(912) << endl;
	cout << X.row(912) << endl;
	cout << " predict is " << get<0>(model.predict(X.row(912))) << endl;

	cout << get<11>(model.predict(X.row(912))) << endl; //a5
	cout << get<12>(model.predict(X.row(912))) << endl; //y_hat
	cout << "==============" << endl;

	cout << " test " << endl;
	cout << "answer is " << Y(30001) << endl;
	cout << X.row(30001) << endl;
	cout << " predict is " << get<0>(model.predict(X.row(30001))) << endl;

	cout << get<11>(model.predict(X.row(30001))) << endl; //a5
	cout << get<12>(model.predict(X.row(30001))) << endl; //y_hat
	cout << "==============" << endl;
	//cout << get<10>(result) << endl; //b5
	//cout << "==============" << endl;
	//cout << get<5>(result) << endl; //W5
	//cout << "==============" << endl;
	//cout << get<11>(result) << endl; //a5
	//cout << "==============" << endl;
	//cout << get<11>(result)(8,0) << endl; //a5
	//cout << "==============" << endl;
	//cout << get<11>(result)(8) << endl; //a5

	/*Eigen::MatrixXd test1;
	test1 = Eigen::MatrixXd(3, 2);
	test1(0, 0) = 1; test1(0, 1) = 2; test1(1,0) = 3; test1(1,1) = 4; test1(2,0) = 5; test1(2,1) = 6;

	Eigen::VectorXd test2 = test1.row(1);

	cout << test1.rows() << " " << test1.cols() << endl;
	cout << "==============" << endl;
	cout << test1 << endl;
	cout << "==============" << endl;
	cout << test2.rows() << " " << test2.cols() << endl;
	cout << "==============" << endl;
	cout << test2 << endl;*/





	


    return 0;
}

