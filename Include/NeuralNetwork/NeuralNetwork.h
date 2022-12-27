//
// Created by alexander on 19-08-22.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <vector>

#include "Layer.h"
#include "../../Include/Utilities/Functions.h"

typedef std::vector<double> vec;
typedef std::vector<std::vector<double>> matrix;

struct data{
    vec input;
    vec output;
};

class NeuralNetwork {
   public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(const std::vector<int>& widths, functions::Activation activationFunction_);

    void addLayers(std::vector<int> widths);

    [[nodiscard]] std::vector<double> getOutput(const std::vector<double>& input) const;

    [[nodiscard]] double calculateCost(const std::vector<double>& input, const std::vector<double>& expectedOutput) const;

    [[nodiscard]] std::vector<LayerOutput> calculateLayerOutputs(const std::vector<double>& input) const;

    void gradientDescent(const std::vector<data>& trainingData, int epochs, double learningRate);

    void updateWeightsAndBiases(const std::vector<matrix>& weightsUpdates, const matrix& biasUpdates);

    void mutate();

    void storeOutput(std::pair<double, double> domain) const;

    [[nodiscard]] std::pair<std::vector<matrix>, matrix>
    backprop(const std::vector<double>& input, const std::vector<double>& expectedOutput);

private:
    std::vector<Layer> layers;
    functions::Activation activationFunction;

};

#endif  // NEURALNETWORK_NEURALNETWORK_H
