//
// Created by alexander on 03-10-22.
//
//

#ifndef NEURALNETWORK_FUNCTIONS_H
#define NEURALNETWORK_FUNCTIONS_H

#include <cmath>
#include <vector>

namespace functions{
    double sigmoid(double x);

    std::vector<double> sigmoid(const std::vector<double>& input);

    double mse(const std::vector<double>& x, const std::vector<double>& y);
}

#endif //NEURALNETWORK_FUNCTIONS_H

