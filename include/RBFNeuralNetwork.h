#pragma once
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

// RBF神经网络类
class RBFNeuralNetwork 
{
public:
    // 构造函数，初始化中心数和标准差
    RBFNeuralNetwork(int num_centers, double sigma);
    // 训练函数，使用输入X和目标Y进行训练，指定训练轮数和学习率
    void train(const std::vector<VectorXd> &X, const std::vector<VectorXd> &Y, int epochs, double learning_rate);
    // 预测函数，根据输入进行预测
    VectorXd predict(const VectorXd &input);

private:
    // 高斯函数，计算点x与中心的高斯距离
    static double gaussian(const VectorXd &x, const VectorXd &center, double sigma);
    // 高斯函数的导数
    static VectorXd gaussian_derivative(const VectorXd &x, const VectorXd &center, double sigma); 
    
    MatrixXd centers_; // 中心点矩阵
    VectorXd sigmas_;  // 标准差向量
    MatrixXd W_;      // 输出层权重矩阵
    int num_centers_;  // 中心数量
    double sigma_;     // 标准差
};