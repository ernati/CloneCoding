#pragma once
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <math.h>
#include <tuple>
#include <random>
#include <vector>



class logistic_regression_model {

public :
	Eigen::Matrix< float, 2, 1 > w;
	float b;

	logistic_regression_model() {
		//난수 준비
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(0, 10000);

		w << ( (float)dis(gen) - 5000.0 ) / 5000.0, ((float)dis(gen) - 5000.0) / 5000.0;
		b = ((float)dis(gen) - 5000.0) / 5000.0;

		//std::cout << "초기값 : " << "w0 : " << w(0) << " " << "w1 : " << w(1) << " " << "b : " << b << std::endl;
	}

	float sigmoid(float z) {
		return 1.0 / (1 + exp(-z));
	}

	//predict 함수 제대로 작동 확인
	float predict(std::tuple<float, float> x) {
		//self.w[0] * x[0]
		float z = w(0) * std::get<0>(x) + w(1) * std::get<1>(x) + b;
		float a = sigmoid(z);
		return a;
	}

	void update_parameter(float w0, float w1, float b0) {
		w(0) = w0;
		w(1) = w1;
		b = b0;
		//std::cout << "w0 : " << w(0) << " " << "w1 : " << w(1) << " " << "b : " << b << std::endl;
	}
};

float train(Eigen::Matrix< std::tuple<float, float>, 4, 1 > X, Eigen::Matrix<float, 4, 1> Y, logistic_regression_model& model, float lr) {
	float dw0 = 0.0;
	float dw1 = 0.0;
	float db = 0.0;
	float m = 4.0; //m = len(X)
	float cost = 0.0;

	for (int i = 0; i < 4; i++) {
		float a = model.predict( X(i) );

		if (Y(i) == 1.0) {
			cost -= log(a);
		}
		else {
			cost -= log(1 - a);
		}

		dw0 += (a - Y(i)) * std::get<0>( X(i) );
		dw1 += (a - Y(i)) * std::get<1>( X(i) );
		db += (a - Y(i));
		//std::cout << dw0 << std::endl;
	}

	cost /= m;
	float w0 = model.w(0) - lr * dw0 / m;
	float w1 = model.w(1) -  lr * dw1 / m;
	float b = model.b -  lr * db / m;
	model.update_parameter(w0, w1, b);

	return cost;
}

int main() {
	//model 선언
	logistic_regression_model model = logistic_regression_model();

	//값들 초기화
	std::tuple<float,float> tuplevalue1 = std::make_tuple(0.0, 0.0);
	std::tuple<float, float> tuplevalue2 = std::make_tuple(1.0, 0.0);
	std::tuple<float, float> tuplevalue3 = std::make_tuple(0.0, 1.0);
	std::tuple<float, float> tuplevalue4 = std::make_tuple(1.0, 1.0);

	//입력값 및 라벨값 선언 및 초기화
	Eigen::Matrix< std::tuple<float, float>, 4, 1 > X{ tuplevalue1, tuplevalue2, tuplevalue3, tuplevalue4 };
	Eigen::Matrix<float, 4, 1> Y_AND{ 0,0,0,1 };
	Eigen::Matrix<float, 4, 1> Y_OR{ 0,1,1,1 };
	Eigen::Matrix<float, 4, 1> Y_XOR{ 0,1,1,0 };

	//model.predict( std::make_tuple( 0.647, 0.985 ) );

	std::vector<float> losses;
	
	for (int i = 0; i < 10000; i++) {
		float cost1 = train(X, Y_OR, model, 0.1);
		losses.push_back(cost1);
		if (i % 100 == 0) {
			std::cout << " epoch : " << i << ", loss : " << cost1 << std::endl;
		}

	}
	//model.predict( (0,0) )
	std::cout << "model.predict(0,0) is " << model.predict(tuplevalue1) << std::endl;
	//model.predict( (0,1) )
	std::cout << "model.predict(0,1) is " << model.predict(tuplevalue3) << std::endl;
	//model.predict( (1,0) )
	std::cout << "model.predict(1,0) is " << model.predict(tuplevalue2) << std::endl;
	//model.predict( (1,1) )
	std::cout << "model.predict(1,1) is " << model.predict(tuplevalue4) << std::endl;
}





