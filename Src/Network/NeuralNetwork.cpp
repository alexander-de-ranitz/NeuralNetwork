//
// Created by alexander on 19-08-22.
//
#include "../../Include/NeuralNetwork/NeuralNetwork.h"
#include "../../Include/Utilities/Functions.h"

#include <stdexcept>

std::vector<double> NeuralNetwork::getOutput(std::vector<double> input) const {
    for (auto& layer : layers) {
        input = layer.calculateOutput(input);
    }
    return input;
}

void NeuralNetwork::addLayers(std::vector<int> widths) {
    if (std::any_of(widths.begin(), widths.end(), [](auto x) { return x < 1; })) throw(std::invalid_argument("Trying to create layer with width < 1!"));

    for (int i = 0; i < widths.size() - 1; ++i) {
        layers.emplace_back(widths.at(i), widths.at(i + 1));
    }
}

double NeuralNetwork::calculateCost(const std::vector<double>& input, const std::vector<double>& expectedOutput) const {
    auto actualOutput = getOutput(input);
    return functions::mse(expectedOutput, actualOutput);
}

void NeuralNetwork::mutate() {
    for (auto& layer : layers) layer.mutateWeightsAndBiases();
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& widths) {
    if (std::any_of(widths.begin(), widths.end(), [](auto x) { return x < 1; })) throw(std::invalid_argument("Trying to create layer with width < 1!"));
    addLayers(widths);
}
