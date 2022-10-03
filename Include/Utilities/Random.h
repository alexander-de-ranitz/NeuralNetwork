//
// Created by alexander on 18-08-22.
//

#ifndef NEURALNETWORK_RANDOM_H
#define NEURALNETWORK_RANDOM_H

#include "random"

class Random {
   public:
    static double getRandDoubleUniform(double min, double max) { return getRandDoubleUniform(min, max, 1).at(0); }

    static std::vector<double> getRandDoubleUniform(double min, double max, int number) {
        static std::random_device rng;
        static std::default_random_engine e1(rng());
        static std::uniform_real_distribution<double> uniform_dist(0, 1.0);

        std::vector<double> output;
        output.reserve(number);
        for (int i = 0; i < number; ++i) {
            output.emplace_back(uniform_dist(e1) * (max - min) + min);
        }
        return output;
    }

    static double getRandDoubleNorm() { return getRandDoubleNorm(1).at(0); }

    static std::vector<double> getRandDoubleNorm(int number) {
        static std::random_device rng;
        static std::default_random_engine e1(rng());
        static std::normal_distribution<double> normalDistribution(0, 1.0);

        std::vector<double> output;
        output.reserve(number);
        for (int i = 0; i < number; ++i) {
            output.emplace_back(normalDistribution(e1));
        }
        return output;
    }
};

#endif  // NEURALNETWORK_RANDOM_H
