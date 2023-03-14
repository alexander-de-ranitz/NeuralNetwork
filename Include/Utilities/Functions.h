//
// Created by alexander on 03-10-22.
//
//

#ifndef NEURALNETWORK_FUNCTIONS_H
#define NEURALNETWORK_FUNCTIONS_H

#include <cmath>
#include <vector>

namespace functions{
    enum class Activation{
        SIGMOID,
        RELU,
        TANH,
        LINEAR
    };

    std::vector<double> activation(Activation function, std::vector<double>& input);

    std::vector<double> activation_derivative(Activation function, std::vector<double>& input);

    double sigmoid(double x);
    std::vector<double> sigmoid(const std::vector<double>& input);

    double sigmoid_prime(double x);
    std::vector<double> sigmoid_prime(const std::vector<double>& input);

    double tanh(double x);
    std::vector<double> tanh(const std::vector<double>& input);

    double tanh_prime(double x);
    std::vector<double> tanh_prime(const std::vector<double>& input);

    double relu(double x);
    std::vector<double> relu(const std::vector<double>& input);

    double relu_prime(double x);
    std::vector<double> relu_prime(const std::vector<double>& input);

    std::vector<double> softmax(const std::vector<double>& input);

    double categoricalCrossEntropy(const std::vector<double>& actual, const std::vector<double>& expected);

    std::vector<double> diff_elem(const std::vector<double>& x, const std::vector<double>& y);

    std::vector<double> product_elem(const std::vector<double>& x, const std::vector<double>& y);

    std::vector<double> scaleByFactor(std::vector<double> vector, double scalar);

    template <typename T>
    std::vector<T> scaleByFactor(std::vector<T> vector, double scalar){
        for (auto& x : vector)
            x = scaleByFactor(x, scalar);
        return vector;
    }

    std::vector<std::vector<double>> outerProduct(const std::vector<double>& x, const std::vector<double>& y);

    double dot(const std::vector<double>& xs, const std::vector<double>& ys);

    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& m);

    /**
     * Multiplies the an MxN matrix with a vector of size M to get a vector of size M
     * The i-th element in the vector will be the sum of all values in the i-th row of the matrix multiplied by the i-th value of the given vector
     * @param mat Vector of vector of doubles of size MxN
     * @param vec Vector of doubles of size N
     * @return vector of size M
     */
    std::vector<double> matVecMult(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec);

    double add(double x, double y);

    /// TODO: move to CPP file
    template <typename T>
    std::vector<T> add(std::vector<T> xs, std::vector<T> ys){
        for (int i = 0; i < xs.size(); ++i) {
            xs[i] = add(xs[i], ys[i]);
        }
        return xs;
    }


    double mse(const std::vector<double>& x, const std::vector<double>& y);
}

#endif //NEURALNETWORK_FUNCTIONS_H

