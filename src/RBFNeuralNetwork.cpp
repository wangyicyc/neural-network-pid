#include "RBFNeuralNetwork.h"
#include <iostream>

// RBF神经网络类
RBFNeuralNetwork::RBFNeuralNetwork(int num_centers, double sigma)
    : num_centers_(num_centers), sigma_(sigma) 
{
    centers_.resize(num_centers_, 3); // 假设输入维度为3 (P, I, D)
    // setConstant 是 Eigen 库中的一种成员函数，用于将矩阵或数组的所有元素设置为指定的常数值，可以快速初始化或者重置矩阵/数组的内容
    sigmas_.setConstant(sigma_);

    W_.resize(1, num_centers_); // 假设输出维度为1 (控制信号)

    // 初始化中心点（可以使用K-means或其他方法）
    for (int i = 0; i < num_centers_; ++i) 
    {
        centers_.row(i) << (rand() % 100) / 100.0, (rand() % 100) / 100.0, (rand() % 100) / 100.0;
    }
}

VectorXd RBFNeuralNetwork::gaussian_derivative(const VectorXd &x, const VectorXd &center, double sigma) 
{
    // 高斯基函数的导数（梯度）
    double g = gaussian(x, center, sigma);
    return ((x - center) / (sigma * sigma)).array() * g;
}

// 计算高斯核函数
double RBFNeuralNetwork::gaussian(const VectorXd &x, const VectorXd &center, double sigma) 
{
    double norm_squared = (x - center).squaredNorm();
    return std::exp(-norm_squared / (2 * sigma * sigma));
}

// 进行预测
VectorXd RBFNeuralNetwork::predict(const VectorXd &input) 
{
    VectorXd hidden(num_centers_);
    for (int i = 0; i < num_centers_; ++i)
    {
        // 计算每个中心点的高斯核函数值
        // 前向传播
        hidden(i) = gaussian(input, centers_.row(i), sigmas_(i));
    }
    // 线性组合隐藏层输出
    return W_ * hidden;
}

// 训练网络
void RBFNeuralNetwork::train(const std::vector<VectorXd> &X, const std::vector<VectorXd> &Y, int epochs, double learning_rate) 
{
    for (int epoch = 0; epoch < epochs; ++epoch) 
    {
        for (size_t i = 0; i < X.size(); ++i) 
        {
            // 前向传播
            // 前向传播过程中，输入通过高斯基函数计算得到隐藏层的激活值 hidden。
            // hidden 是一个 VectorXd，其长度等于隐藏层神经元的数量,这里等于中心点的数量 num_centers_。
            VectorXd hidden(num_centers_);
            for (int j = 0; j < num_centers_; ++j)
            {
                hidden(j) = gaussian(X[i], centers_.row(j), sigmas_(j));
            }
            VectorXd prediction = predict(X[i]);

            // 计算误差
            VectorXd error = Y[i] - prediction;

            // 反向传播
            // 更新输出层权重
            // error(0) 表示误差向量的第一个元素（假设输出是一维的）。
            // hidden.transpose() 表示将隐藏层输出转置，方便与权重矩阵相乘。
            // 隐藏层激活值反映了输入对各个隐藏层神经元的影响。转置后，它可以与误差项相乘，生成权重更新的梯度。
            W_ += learning_rate * error(0) * hidden.transpose();

            // 如果需要调整中心点和标准差，可以在这里添加相应的代码
            // 注意：RBF 网络通常不直接调整中心点和标准差，而是使用聚类等方法预先确定:
            // 1. 聚类：将输入数据集划分为 K 个簇，每个簇对应一个中心点
            // 2. 基于 K-means 的迭代优化算法：在每轮迭代中，根据当前的中心点重新计算标准差，并更新中心点
            // 3. 基于 EM 算法的迭代优化算法：在每轮迭代中，根据当前的中心点重新计算标准差，并更新中心点
            
            // 计算隐藏层梯度（这里假设只有一维输出）
            VectorXd hidden_gradient = W_.transpose() * error;
            // 更新中心点和标准差（如果需要）
            for (int j = 0; j < num_centers_; ++j) 
            {
                // 更新中心点
                centers_.row(j) += learning_rate * hidden_gradient(j) * gaussian_derivative(X[i], centers_.row(j), sigmas_(j));
                // 更新标准差（可选）
                // sigmas_(j) += learning_rate * hidden_gradient(j) * /* some derivative */;
            }
        }
    }
}