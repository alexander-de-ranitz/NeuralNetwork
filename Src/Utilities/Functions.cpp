//
// Created by alexander on 03-10-22.
//
#include <stdexcept>

#include "../../Include/Utilities/Functions.h"

namespace functions{
    double sigmoid(double x){
        return 1.0/(std::pow(M_E, -x));
    }

    std::vector<double> sigmoid(const std::vector<double>& input){
        std::vector<double> output;
        output.reserve(input.size());
        for (auto& x : input){
            output.emplace_back(sigmoid(x));
        }
        return output;
    }

    double mse(const std::vector<double>& x, const std::vector<double>& y){
        if (x.size() != y.size())
            throw(std::invalid_argument("Vector of size " + std::to_string(x.size()) + " does not match with vector of size " + std::to_string(y.size())));

        double error = 0;
        for (int i = 0; i < x.size(); ++i) {
            error += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return error / static_cast<double>(x.size());
    }
}