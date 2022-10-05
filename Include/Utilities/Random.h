//
// Created by alexander on 18-08-22.
//

#ifndef NEURALNETWORK_RANDOM_H
#define NEURALNETWORK_RANDOM_H

#include <random>

class Random {
   public:
    static double getRandDoubleUniform(double min, double max);

    static std::vector<double> getRandDoubleUniform(double min, double max, int number);

    static double getRandDoubleNorm();

    static std::vector<double> getRandDoubleNorm(int number);
};

#endif  // NEURALNETWORK_RANDOM_H
