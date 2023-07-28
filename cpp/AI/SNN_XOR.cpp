#pragma once
#include <iostream>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <math.h>
#include <tuple>
#include <random>
#include <vector>
#include <tgmath.h>


class shallow_neural_network {
public:

	int num_input_features;
	int num_hiddens;

	//동적 행렬 선언
	Eigen::MatrixXd W1;
	Eigen::VectorXd b1;
	Eigen::VectorXd W2;
	float b2;

	//생성자 - 기본 형태 
	shallow_neural_network() {

	}

	shallow_neural_network( int num_input_feature, int num_hidden ) {
		//난수 준비
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(0, 1000);

		this->num_input_features = num_input_feature;
		this->num_hiddens = num_hidden;

		//동적 행렬 사이즈 조정 및 초기화
		W1 = Eigen::MatrixXd(num_hiddens, num_input_features); // 3 x 2
		b1 = Eigen::VectorXd(num_hiddens); // 3 x 1
		W2 = Eigen::VectorXd(num_hiddens); // 3 x 1
		b2 = ((float)dist(gen) / 1000.0);

		W1.setZero();
		b1.setZero();
		W2.setZero();

		//W1에 random값들 더해주기
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				W1(i,j) = W1(i,j) + ((float)dist(gen) - 500) / 1000.0;
			}
		}

		//b1에 random값들 더해주기
		for (int i = 0; i < 3; i++) {
			b1(i) = b1(i) + ((float)dist(gen) - 500) / 1000.0;
		}

		//W2에 random값들 더해주기
		for (int i = 0; i < 3; i++) {
			W2(i) = W2(i) + ((float)dist(gen) - 500) / 1000.0;
		}
	}

	float sigmoid(float z) {
		return 1.0 / (1 + exp(-z));
	}

	std::tuple< float, Eigen::VectorXd, Eigen::VectorXd, float, float > predict(Eigen::VectorXd x) {
		// z1 = W1 * x-- > (num_hiddens, 1)
		Eigen::VectorXd z1(num_hiddens);
		Eigen::VectorXd a1(num_hiddens);
		float z2, a2;
		z1 = W1 * x + b1;
		for (int i = 0; i < num_hiddens; i++) {
			a1(i) = tanh(z1(i));
		}

		z2 = W2.dot(a1) + b2;
		a2 = sigmoid(z2);

		std::tuple<float, Eigen::VectorXd, Eigen::VectorXd, float, float> result = std::make_tuple(a2, z1, a1, z2, a2);

		return result;
	}

};

float train(Eigen::MatrixXd X, Eigen::VectorXd Y, shallow_neural_network& model, float lr, int N) {
	//동적 행렬 선언
	Eigen::MatrixXd dW1;
	Eigen::VectorXd db1;
	Eigen::VectorXd dW2;
	float db2;

	float m = (float)N;
	float cost = 0.0;

	//동적 행렬 사이즈 조정 및 초기화
	dW1 = Eigen::MatrixXd(model.num_hiddens, model.num_input_features); // 3 x 2
	dW1.setZero();
	db1 = Eigen::VectorXd(model.num_hiddens); // 2 x 1
	db1.setZero();
	dW2 = Eigen::VectorXd(model.num_hiddens); // 2 x 1
	dW2.setZero();
	db2 = 0.0;

	for (int i = 0; i < N; i++) {
		//std::cout << " train cycle " << i << " start!" << std::endl;

		//predict 입력으로 넣을 x 생성
		Eigen::VectorXd x = Eigen::VectorXd(model.num_input_features);
		x(0) = X(i, 0); 
		x(1) = X(i, 1);

		//std::cout << "x is " << x << std::endl;

		std::tuple<float, Eigen::VectorXd, Eigen::VectorXd, float, float> predict = model.predict(x);
		float a2 = std::get<0>(predict); //float
		//Eigen::VectorXd a1 = std::get<2>(predict); //num_hiddens 크기의 vector
		Eigen::VectorXd a1 = Eigen::VectorXd(model.num_hiddens);
		Eigen::VectorXd e1 = Eigen::VectorXd(model.num_hiddens);
		Eigen::VectorXd one_vector = Eigen::VectorXd(model.num_hiddens);
		one_vector.setOnes();

		a1(0) = std::get<2>(predict)(0);
		a1(1) = std::get<2>(predict)(1);
		a1(2) = std::get<2>(predict)(2);

		if (Y(i) == 1) {
			cost -= log(a2);
		}
		else {
			cost -= log(1 - a2);
		}

		/*std::cout << " db2_before is " << db2 << std::endl;
		std::cout << " a2 is " << a2 << std::endl;
		std::cout << " Y(i) is " << Y(i) << std::endl;*/

		db2 = db2 + (a2 - Y(i));

		/*std::cout << " db2_after is " << db2 << std::endl;

		std::cout << " dW2_before is " << dW2 << std::endl;*/
		dW2 = dW2 + (a2 - Y(i)) * a1;

		/*std::cout << " a1 is " << std::get<2>(predict) << std::endl;
		std::cout << " dW2_after is " << dW2 << std::endl;*/

		e1 = one_vector - (a1 * a1);

		db1 = db1 + (a2 - Y(i)) * dW2 + e1;

		dW1 = dW1 + (a2 - Y(i)) * (dW2 * e1) * x.transpose();

		//std::cout << dW1 << std::endl;

	}

	cost /= m;
	model.W1 -= lr * dW1 / m;
	model.b1 -= lr * db1 / m;
	model.W2 -= lr * dW2 / m;
	model.b2 -= lr * db2 / m;

	/*std::cout << " dW1 is " << std::endl << dW1 << std::endl;
	std::cout << " db1 is " << std::endl << db1 << std::endl;
	std::cout << " dW2 is " << std::endl << dW2 << std::endl;
	std::cout << " db2 is " << std::endl << db2 << std::endl;*/

	return cost;
	


}

int main(void) {
	//x_seeds, y_seeds 선언
	Eigen::MatrixXd x_seeds(4, 2);
	Eigen::VectorXd y_seeds(4);
	x_seeds(0, 0) = 0; x_seeds(1, 0) = 1; x_seeds(2, 0) = 0; x_seeds(3, 0) = 1;
	x_seeds(0, 1) = 0; x_seeds(1, 1) = 0; x_seeds(2, 1) = 1; x_seeds(3, 1) = 1;
	y_seeds(0) = 0; y_seeds(1) = 1; y_seeds(2) = 1; y_seeds(3) = 0;

	//idxs
	int N = 1000;

	//난수 준비
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(0, 3); // 0부터 3까지를 random으로 생성
	std::uniform_int_distribution<int> dist(0, 1000);

	std::vector<int> idxs;
	for (int i = 0; i < N; i++) {
		idxs.push_back( dis(gen) );
	}

	//X,Y 준비
	Eigen::MatrixXd X(N, 2);
	Eigen::VectorXd Y(N);
	//X(i,0) = x_seeds( idxs[i],0 ) X(i,1) = x_seeds( idxs[i],1 ) ...
	for (int i = 0; i < N; i++) {
		X(i, 0) = x_seeds(idxs[i], 0);
		X(i, 1) = x_seeds(idxs[i], 1);
		Y(i) = y_seeds(idxs[i]);
	}

	//X에 random값들 더해주기
	for (int i = 0; i < N; i++) {
		X(i, 0) = X(i, 0) + ((float)dist(gen) - 500) / 1000.0;
		X(i, 1) = X(i, 1) + ((float)dist(gen) - 500) / 1000.0;
	}



	shallow_neural_network model = shallow_neural_network(2, 3);

	float cost = 0.0;

	//train
	for (int i = 0; i < 100; i++) {
		cost = train(X, Y, model, 0.1, N);
		if (i % 10 == 0) {
			std::cout << "epoch : " << i << " // cost : " << cost << std::endl;
		}
	}

	std::cout << std::endl << "=========================================" << std::endl;

	//결과 확인
	Eigen::VectorXd test1(2); test1 << 1, 1;
	Eigen::VectorXd test2(2); test2 << 1, 0;
	Eigen::VectorXd test3(2); test3 << 0, 1;
	Eigen::VectorXd test4(2); test4 << 0, 0;


	std::cout << " predict (1,1) ---> " << std::get<0>(model.predict(test1)) << std::endl;
	std::cout << " predict (1,0) ---> " << std::get<0>(model.predict(test2)) << std::endl;
	std::cout << " predict (0,1) ---> " << std::get<0>(model.predict(test3)) << std::endl;
	std::cout << " predict (0,0) ---> " << std::get<0>(model.predict(test4)) << std::endl;
	
	


	
}