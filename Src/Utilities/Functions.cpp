//
// Created by alexander on 03-10-22.
//
#include <stdexcept>

#include "../../Include/Utilities/Functions.h"

namespace functions{

    /// Activation Functions
    std::vector<double> activation(Activation function, std::vector<double>& input){
        switch (function) {
            case Activation::SIGMOID:
                return sigmoid(input);
            case Activation::RELU:
                return relu(input);
            case Activation::LINEAR:
                return input;
        }
    }

    double sigmoid(double x){
        return 1.0/(1 + std::pow(M_E, -x));
    }

    std::vector<double> sigmoid(const std::vector<double>& input){
        std::vector<double> output;
        output.reserve(input.size());
        for (auto& x : input){
            output.emplace_back(sigmoid(x));
        }
        return output;
    }

    double relu(double x){
        return x > 0 ? x : 0;
    }

    std::vector<double> relu(const std::vector<double>& input){
        std::vector<double> output;
        output.reserve(input.size());
        for (auto& x : input){
            output.emplace_back(relu(x));
        }
        return output;
    }

    /// Derivatives
    std::vector<double> activation_derivative(Activation function, std::vector<double>& input){
        switch (function) {
            case Activation::SIGMOID:
                return sigmoid_prime(input);
            case Activation::RELU:
                return relu_prime(input);
            case Activation::LINEAR:
                return std::vector<double>(input.size(), 1.0);
        }
    }

    double sigmoid_prime(double x){
        return sigmoid(x) * (1 - sigmoid(x));
    }

    std::vector<double> sigmoid_prime(const std::vector<double>& input){
        std::vector<double> output;
        output.reserve(input.size());
        for (auto& x : input){
            output.emplace_back(sigmoid_prime(x));
        }
        return output;
    }

    double relu_prime(double x){
        return x > 0 ? 1 : 0;
    }

    std::vector<double> relu_prime(const std::vector<double>& input){
        std::vector<double> output;
        output.reserve(input.size());
        for (auto& x : input){
            output.emplace_back(relu_prime(x));
        }
        return output;
    }

    /// Error functions
    double mse(const std::vector<double>& x, const std::vector<double>& y){
        if (x.size() != y.size())
            throw(std::invalid_argument("Vector of size " + std::to_string(x.size()) + " does not match with vector of size " + std::to_string(y.size())));

        double error = 0;
        for (int i = 0; i < x.size(); ++i) {
            error += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return error / static_cast<double>(x.size());
    }

    /// Utility functions
    std::vector<double> diff_elem(const std::vector<double>& xs, const std::vector<double>& ys){
        auto result = xs;
        for (int i = 0; i < xs.size(); ++i) {
            result[i] -= ys[i];
        }
        return result;
    }

    std::vector<double> product_elem(const std::vector<double>& xs, const std::vector<double>& ys){
        auto result = xs;
        for (int i = 0; i < xs.size(); ++i) {
            result[i] *= ys[i];
        }
        return result;
    }


    std::vector<double> scaleByFactor(std::vector<double> vector, double scalar){
        for (auto& x : vector)
            x *= scalar;
        return vector;
    }


    std::vector<std::vector<double>> outerProduct(const std::vector<double>& x, const std::vector<double>& y){
        std::vector<std::vector<double>> result;
        result.reserve(x.size());
        for (double i : x) {
            result.emplace_back(scaleByFactor(y, i));
        }
        return result;
    }

    double dot(const std::vector<double>& xs, const std::vector<double>& ys){
        double result = 0;
        for (int i = 0; i < xs.size(); ++i) {
            result += xs[i] * ys[i];
        }
        return result;
    }

    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& m){
        std::vector<std::vector<double>> result(m[0].size(), std::vector<double>(m.size()));

        for (int i = 0; i < m.size(); ++i) {
            for (int j = 0; j < m[0].size(); ++j) {
                result[j][i] = m[i][j];
            }
        }

        return result;
    }

    std::vector<double> matVecMult(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec){
        std::vector<double> result;
        result.reserve(mat.size());

        for (int i = 0; i < mat.size(); ++i) {
            double total = 0.0;
            for (int j = 0; j < mat[i].size(); ++j) {
                total += mat[i][j] * vec[j];
            }
            result.emplace_back(total);
        }
        return result;
    }

    double add(double x, double y){
        return x + y;
    }
}