#pragma once
#include "RBFNeuralNetwork.h"

class RBFPIDController 
{
public:
    // 构造函数
    RBFPIDController(double kp, double ki, double kd, int num_centers, double sigma)
    : kp_(kp), ki_(ki), kd_(kd), integral_(0), prev_error_(0),
      rbf_network_(num_centers, sigma) {}
    // 更新函数
    double update(double setpoint, double measurement, double dt);
    // 调整参数
    void adjust_parameters(const VectorXd &input);

private:
    double kp_, ki_, kd_;
    double integral_, prev_error_;
    RBFNeuralNetwork rbf_network_;
};
