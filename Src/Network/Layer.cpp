//
// Created by alexander on 19-08-22.
//

#include "../../Include/NeuralNetwork/Layer.h"

#include <cmath>
#include <stdexcept>

#include "../../Include/Utilities/Functions.h"
#include "../../Include/Utilities/Random.h"

Layer::Layer(int inputSize_, int outputSize_) : inputSize(inputSize_), outputSize(outputSize_) {
    // For each neuron, set bias to 0
    biases = std::vector<double>(outputSize_, 0);

    // For each neuron, initialise a vector of 0-weights equal in size to the previous layer
    weightsMatrix.reserve(outputSize_);
    for (int i = 0; i < outputSize_; ++i) {
        // Randomly initialise weights according to Xavier method
        weightsMatrix.emplace_back(Random::getRandDoubleUniform(-sqrt(inputSize_), sqrt(inputSize_), inputSize_));
    }
}

std::vector<double> Layer::calculateOutput(const std::vector<double>& input) const {
    if (input.size() != inputSize) {
        throw(std::invalid_argument("Input size of " + std::to_string(input.size()) + " does not match expected size of " + std::to_string(inputSize)));
    }

    auto output = biases;
    // Loop over each neuron in this layer
    for (int i = 0; i < outputSize; ++i) {
        // Loop over each input
        for (int j = 0; j < inputSize; ++j) {
            // Add the weighted input to the output for this neuron
            output.at(i) += weightsMatrix[i][j] * input[j];
        }
    }

    return functions::sigmoid(output);
}

void Layer::mutateBiases() {
    // These are magic numbers, proper tuning can probably improve performance
    constexpr double BIAS_MUTATION_FACTOR = 0.01;
    constexpr double MUTATION_CHANCE = 0.75;
    for (auto& bias : biases) {
        if (Random::getRandDoubleUniform(0.0, 1.0) > MUTATION_CHANCE) bias += Random::getRandDoubleNorm() * BIAS_MUTATION_FACTOR;
    }
}

void Layer::mutateWeights() {
    // These are magic numbers, proper tuning can probably improve performance
    constexpr double WEIGHT_MUTATION_FACTOR = 0.1;
    constexpr double MUTATION_CHANCE = 0.75;
    for (auto& weightVector : weightsMatrix) {
        for (auto& weight : weightVector)
            if (Random::getRandDoubleUniform(0.0, 1.0) > MUTATION_CHANCE) weight += Random::getRandDoubleNorm() * WEIGHT_MUTATION_FACTOR;
    }
}

void Layer::mutateWeightsAndBiases() {
    mutateWeights();
    mutateBiases();
}
