#include "RBFPIDController.h"
#include <cmath>

double RBFPIDController::update(double setpoint, double measurement, double dt) {
    double error = setpoint - measurement;
    integral_ += error * dt;
    double derivative = (error - prev_error_) / dt;

    // 构建输入向量用于调整 PID 参数
    VectorXd input(3);
    input << error, integral_, derivative;

    // 调整 PID 参数
    adjust_parameters(input);

    // 计算控制信号
    double control_signal = kp_ * error + ki_ * integral_ + kd_ * derivative;

    prev_error_ = error;

    return control_signal;
}

void RBFPIDController::adjust_parameters(const VectorXd &input) 
{
    // 使用 RBF 网络预测新的 PID 参数
    VectorXd new_parameters = rbf_network_.predict(input);

    // 更新 PID 参数
    kp_ = new_parameters(0);
    ki_ = new_parameters(1);
    kd_ = new_parameters(2);

    // 可以添加一些限制条件以确保参数在合理范围内
    kp_ = std::max(0.0, std::min(kp_, 10.0)); // 例如，限制在 [0, 10] 之间
    ki_ = std::max(0.0, std::min(ki_, 1.0));
    kd_ = std::max(0.0, std::min(kd_, 5.0));
}