#ifndef MATH_DEFINITIONS
#define MATH_DEFINITIONS

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <math.h>
#include <tuple>

//입력값 및 라벨값 선언 및 초기화
Eigen::Matrix< tuple<float,float>, 4,1 > X;
Eigen::VectorXd Y_AND(4);
Eigen::VectorXd Y_OR(4);
Eigen::VectorXd Y_XOR(4);

//X값 초기화
auto tuplevalue1 = std::make_tuple(0.0, 0.0);
auto tuplevalue2 = std::make_tuple(1.0, 0.0);
auto tuplevalue3 = std::make_tuple(0.0, 1.0);
auto tuplevalue4 = std::make_tuple(1.0, 1.0);
X(0,0) = tuplevalue1;
X(0,1) = tuplevalue2;
X(0,2) = tuplevalue3;
X(0,3) = tuplevalue4;

//라벨값 초기화
Y_AND(0) = 0;
Y_AND(1) = 0;
Y_AND(2) = 0;
Y_AND(3) = 1;

Y_OR(0) = 0;
Y_OR(1) = 1;
Y_OR(2) = 1;
Y_OR(3) = 1;

Y_XOR(0) = 0;
Y_XOR(1) = 1;
Y_XOR(2) = 1;
Y_XOR(3) = 0;

