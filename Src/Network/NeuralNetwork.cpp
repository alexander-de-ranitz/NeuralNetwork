//
// Created by alexander on 19-08-22.
//
#include "../../Include/NeuralNetwork/NeuralNetwork.h"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <fstream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& widths, functions::Activation activationFunction_) {
    if (std::any_of(widths.begin(), widths.end(), [](auto x) { return x < 1; })) throw(std::invalid_argument("Trying to create layer with width < 1!"));
    activationFunction = activationFunction_;
    addLayers(widths);
}

std::vector<double> NeuralNetwork::getOutput(const std::vector<double>& input) const {
    auto output = input;
    for (auto& layer : layers) {
        output = layer.calculateOutput(output);
    }
    return output;
}

void NeuralNetwork::addLayers(std::vector<int> widths) {
    if (std::any_of(widths.begin(), widths.end(), [](auto x) { return x < 1; })) throw(std::invalid_argument("Trying to create layer with width < 1!"));

    for (int i = 0; i < widths.size() - 2; ++i) {
        layers.emplace_back(widths.at(i), widths.at(i + 1), LayerType::hidden, activationFunction);
    }

    layers.emplace_back(widths.at(widths.size() - 2), widths.back(), LayerType::output);
}

double NeuralNetwork::calculateCost(const std::vector<double>& input, const std::vector<double>& expectedOutput) const {
    auto actualOutput = getOutput(input);
    return functions::mse(expectedOutput, actualOutput);
}

void NeuralNetwork::mutate() {
    for (auto& layer : layers) layer.mutateWeightsAndBiases();
}

std::pair<std::vector<matrix>, matrix>
NeuralNetwork::backprop(const std::vector<double> &input, const std::vector<double> &expectedOutput) {

    std::vector<std::vector<std::vector<double>>> weightsUpdates(layers.size());
    std::vector<std::vector<double>> biasUpdates(layers.size());

    auto layerOutputs = calculateLayerOutputs(input);
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> totalInputs;
    activations.reserve(layerOutputs.size());
    totalInputs.reserve(layerOutputs.size());

    for (auto& [activation, totalInput] : layerOutputs) {
        activations.emplace_back(activation);
        totalInputs.emplace_back(totalInput);
    }

    activations.insert(activations.begin(), input);

    // dC/dA should be positive when actual > expected, and negative when actual < expected, so dC/dA = actual - expected
    std::vector<double> errors = functions::diff_elem(expectedOutput, activations.back());
    std::vector<double> delta = functions::product_elem(errors, functions::activation_derivative(activationFunction,totalInputs.back()));

    weightsUpdates[layers.size() - 1] = functions::outerProduct(delta, activations.at(activations.size() - 2));
    biasUpdates[layers.size() -1 ] = delta;
    for (int i = layers.size() - 2; i >= 0; --i) {
        // W^(L+1)^T * delta^L+1 * s'(z) | The transposed weightsmatrix of the next layer multiplied by the error being propagated backwards times the sigmoid derivative at z
        //delta = functions::product_elem(functions::matVecMult(functions::transpose(layers[i+1].getWeights()), errors), functions::sigmoid_prime(totalInputs.at(i)));
        auto w = functions::transpose(layers[i+1].getWeights());
        auto x = functions::matVecMult(w, delta);
        auto y = functions::activation_derivative(activationFunction, totalInputs.at(i));
        delta = functions::product_elem(x, y);
        weightsUpdates[i] = functions::outerProduct(delta, activations[i]);
        biasUpdates[i] = delta;
    }

    return {weightsUpdates, biasUpdates};
}

std::vector<LayerOutput> NeuralNetwork::calculateLayerOutputs(const std::vector<double> &input) const {
    std::vector<LayerOutput> layerOutputs;
    layerOutputs.reserve(layers.size());


    auto lastOutput = input;
    for (auto& layer : layers){
        layerOutputs.emplace_back(layer.calculateLayerOutput(lastOutput));
        lastOutput = layerOutputs.back().activations;
    }

    return layerOutputs;
}

void NeuralNetwork::gradientDescent(const std::vector<data>& trainingData, int epochs, double learningRate) {
    for (int i = 0; i < epochs; ++i) {

        // TODO: Initialise to 0s instead of using 1st data point
        auto backpropResult = backprop(trainingData[0].input, trainingData[0].output);
        auto weightsUpdates = backpropResult.first;
        auto biasUpdates = backpropResult.second;

        for (auto& example : trainingData) {
            backpropResult = backprop(example.input, example.output);
            weightsUpdates = functions::add(weightsUpdates, backpropResult.first);
            biasUpdates = functions::add(biasUpdates, backpropResult.second);
        }

        /// Normalise based on training data size
        weightsUpdates = functions::scaleByFactor(weightsUpdates, 1.0/static_cast<double>(trainingData.size()) * learningRate);
        biasUpdates = functions::scaleByFactor(biasUpdates, 1.0/static_cast<double>(trainingData.size()) * learningRate);

        updateWeightsAndBiases(weightsUpdates, biasUpdates);

        std::cout << "====================" << std::endl;
        std::cout << "Epoch " << i << std::endl;

        double fitness = 0;
        for (const auto& d : trainingData)
            fitness += calculateCost(d.input, d.output);
        fitness /= static_cast<double>(trainingData.size());
        std::cout << "Current cost = " << fitness << std::endl;
    }
}

void NeuralNetwork::updateWeightsAndBiases(const std::vector<matrix> &weightsUpdates, const matrix& biasUpdates) {
    for (int i = 0; i < layers.size(); ++i) {
        layers[i].updateWeightsAndBiases(weightsUpdates[i], biasUpdates[i]);
    }
}

void NeuralNetwork::storeOutput(std::pair<double, double> domain) const {
    if (domain.first > domain.second) throw(std::invalid_argument("Domain not properly defined! Domain should range from lowest to highest!"));

    static std::fstream f;

    // Not super elegant, but this ensures that the first time writing to the file, its contents are wiped
    static bool firstOpen = true;
    if (firstOpen){
        f.open("nnData.txt", std::fstream::out | std::fstream::trunc);
        firstOpen = false;
    } else {
        f.open("nnData.txt", std::fstream::out | std::fstream::app);
    }

    auto x = domain.first;
    // Store the output of the given NN on 50 intervals in the given domain
    for (int i = 0; i < 50; i++) {
        auto y = getOutput({x});
        f << y[0] << ",";
        x += (domain.second - domain.first)/50.0;
    }
    f << "\n";
    f.close();
}
