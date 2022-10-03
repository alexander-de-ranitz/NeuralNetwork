//
// Created by alexander on 19-08-22.
//
#include "../../Include/NeuralNetwork/NeuralNetwork.h"

#include <stdexcept>
#include <utility>

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

double NeuralNetwork::calculateCost(std::vector<double> input, std::vector<double> expectedOutput) const {
    auto actualOutput = getOutput(std::move(input));
    double error = 0.0;
    for (int i = 0; i < expectedOutput.size(); ++i) {
        error += (expectedOutput.at(i) - actualOutput.at(i)) * (expectedOutput.at(i) - actualOutput.at(i));
    }
    return error / static_cast<double>(expectedOutput.size());
}

void NeuralNetwork::mutate() {
    for (auto& layer : layers) layer.mutateWeightsAndBiases();
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& widths) {
    if (std::any_of(widths.begin(), widths.end(), [](auto x) { return x < 1; })) throw(std::invalid_argument("Trying to create layer with width < 1!"));
    addLayers(widths);
}
