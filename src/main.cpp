#include "RBFPIDController.h"
#include <iostream>

int main() 
{
    double kp = 1.0, ki = 0.1, kd = 0.05;
    // 这里假设有10个中心点
    int num_centers = 10;
    // 这里假设每个中心点的标准差为1.0
    double sigma = 1.0;

    RBFPIDController pid(kp, ki, kd, num_centers, sigma);

    // 模拟数据
    double setpoint = 1.0;
    double measurement = 0.0;
    double dt = 0.1;

    for (int t = 0; t < 100; ++t) 
    {
        double control_signal = pid.update(setpoint, measurement, dt);
        // 更新测量值（这里假设简单的积分模型）
        measurement += control_signal * dt;

        std::cout << "Time: " << t * dt << ", Control Signal: " << control_signal
                  << ", Measurement: " << measurement << std::endl;
    }

    return 0;
}