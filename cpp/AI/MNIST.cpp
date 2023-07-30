#pragma once

#include "opencv2/opencv.hpp"
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

using namespace std;
using namespace cv;



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

class MLP {
public:
	//W,b 선언 및 초기화
	Eigen::MatrixXd W1, W2, W3, W4, W5;
	Eigen::VectorXd b1, b2, b3, b4, b5;

	MLP() {
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

		//random값 더해주기
		//난수 준비
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0, 1000);

		//W1에 random값들 더해주기
		for (int i = 0; i < 784; i++) {
			for (int j = 0; j < 512; j++) {
				W1(i,j) = W1(i,j) + ((float)dist(gen) - 500) / 1000.0;
			}
		}
		//W2,b1에 random값들 더해주기
		for (int i = 0; i < 512; i++) {
			for (int j = 0; j < 256; j++) {
				W2(i, j) = W2(i, j) + ((float)dist(gen) - 500) / 1000.0;
			}
			b1(i) = b1(i) + ((float)dist(gen) - 500) / 1000.0;
		}
		//W3,b2에 random값들 더해주기
		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 128; j++) {
				W3(i, j) = W3(i, j) + ((float)dist(gen) - 500) / 1000.0;
			}
			b2(i) = b2(i) + ((float)dist(gen) - 500) / 1000.0;
		}
		//W4, b3에 random값들 더해주기
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 64; j++) {
				W4(i, j) = W4(i, j) + ((float)dist(gen) - 500) / 1000.0;
			}
			b3(i) = b3(i) + ((float)dist(gen) - 500) / 1000.0;
		}
		//W5, b4에 random값들 더해주기
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 10; j++) {
				W5(i, j) = W5(i, j) + ((float)dist(gen) - 500) / 1000.0;
			}
			b4(i) = b4(i) + ((float)dist(gen) - 500) / 1000.0;
		}

		//b5에 random 값들 더해주기
		for (int i = 0; i < 10; i++) {
			b5(i) = b5(i) + ((float)dist(gen) - 500) / 1000.0;
		}
	}

	void ReLU( Eigen::MatrixXd& a ) {
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				if (a(i, j) < 0) {
					a(i, j) = 0;
				}
			}
		}
	}

	
	//오호... 이게 맞는 것 같군
	void softmax(Eigen::MatrixXd& a) {
		double sum = 0;
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				sum += exp(a(i, j));
			}
		}
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < a.cols(); j++) {
				a(i, j) = exp(a(i, j)) / sum;
			}
		}
	}

	//return a6, w1,w2,w3,w4,w5,b1,b2,b3,b4,b5
	std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
	Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> predict(Eigen::VectorXd x) {

		Eigen::MatrixXd a1, a2, a3, a4, a5, a6;
		double digit;
		a1 = W1 * x + b1;
		ReLU(a1);
		a2 = W2 * a1 + b2;
		ReLU(a2);
		a3 = W3 * a2 + b3;
		ReLU(a3);
		a4 = W4 * a3 + b4;
		ReLU(a4);
		a5 = W5 * a4 + b5;
		softmax(a5);

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
		cout << digit << endl;*/

		return std::make_tuple(digit, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5);
	}

};

double train(Eigen::MatrixXd X, Eigen::VectorXd Y, MLP& model, float lr, int N) {
	//dW,db 선언 및 초기화
	Eigen::MatrixXd dW1, dW2, dW3, dW4, dW5;
	Eigen::VectorXd db1, db2, db3, db4, db5;
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

	double m = (double)N;
	double cost = 0.0;

	//100장당 1번 update
	for (int epoch = 0; epoch < 600; epoch++) {
		cout << " epoch : " << epoch << endl;

		//한 epoch당 100장씩 train
		for (int i = epoch * 100; i < epoch * 100 + 100; i++) {
			//X의 각 row를 predict
			Eigen::VectorXd x = X.row(i);

			//std::cout << "x is " << x << std::endl;

			//digit, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5 
			std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
				Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> predict = model.predict(x);
			//digit
			double digit = std::get<0>(predict);
			Eigen::VectorXd one_vector_b5 = Eigen::VectorXd(10);
			one_vector_b5.setOnes();


			/*if (Y(i) == 1) {
				cost -= log(a2);
			}
			else {
				cost -= log(1 - a2);
			}*/

			cost -= Y(i) * log(Y(i)) - (1 - digit) * log(1 - digit);

			db5 = db5 + std::get<10>(predict) - one_vector_b5;
			dW5 = dW5 + (digit - Y(i)) * std::get<9>(predict) * std::get<8>(predict) * std::get<7>(predict) * std::get<6>(predict).transpose();
			db4 = db4 + (digit - Y(i)) * std::get<9>(predict) * std::get<8>(predict) * std::get<7>(predict);
			dW4 = dW4 + (digit - Y(i)) * std::get<9>(predict) * std::get<8>(predict) * std::get<6>(predict).transpose();
			db3 = db3 + (digit - Y(i)) * std::get<9>(predict) * std::get<7>(predict);
			dW3 = dW3 + (digit - Y(i)) * std::get<9>(predict) * std::get<6>(predict).transpose();
			db2 = db2 + (digit - Y(i)) * std::get<9>(predict);
			dW2 = dW2 + (digit - Y(i)) * std::get<8>(predict) * std::get<7>(predict) * std::get<6>(predict).transpose();
			db1 = db1 + (digit - Y(i)) * std::get<8>(predict) * std::get<7>(predict);
			dW1 = dW1 + (digit - Y(i)) * std::get<8>(predict) * std::get<6>(predict).transpose();
		}

		//update
		cost /= m;
		model.W1 -= lr * dW1 / m;
		model.b1 -= lr * db1 / m;
		model.W2 -= lr * dW2 / m;
		model.b2 -= lr * db2 / m;
		

	}


	return cost;

}



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

	std::tuple< double, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
		Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> result = model.predict(X.row(0));

	float cost = 0.0;

	cout << endl << "MNIST to Eigen Done" << endl;

	cost = train(X, Y, model, 0.1, DataNum);

	cout << endl << "Training Done" << endl;


	cout << " test " << endl;
	cout << X.row(57) << endl;
	cout << get<0>(model.predict(X.row(57))) << endl;

	//cout << get<0>(result) << endl; //digit
	//cout << "==============" << endl;
	//cout << get<1>(result) << endl; //W1
	//cout << "==============" << endl;
	//cout << get<8>(result) << endl; //b3
	//cout << "==============" << endl;
	//cout << get<8>(result).rows() << endl; //b4

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

