//
// Created by alexander on 19-08-22.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include "../../Include/Utilities/Functions.h"

#include <vector>

struct LayerOutput{
    std::vector<double> activations;
    std::vector<double> totalInput;
};

enum LayerType{
    hidden,
    output
};

class Layer {
   public:
    Layer(int inputSize_, int outputSize_, LayerType layerType_ = LayerType::hidden, functions::Activation activationFunction_ = functions::Activation::LINEAR);

    [[nodiscard]] std::vector<double> calculateOutput(const std::vector<double>& input) const;

    [[nodiscard]] LayerOutput calculateLayerOutput(const std::vector<double>& input) const;

    void updateWeightsAndBiases(const std::vector<std::vector<double>>& weightUpdates, const std::vector<double>& biasUpdates);

    void mutateBiases();

    void mutateWeights();

    void mutateWeightsAndBiases();

    std::vector<std::vector<double>> getWeights();

private:
    LayerType layerType;
    const int inputSize;
    const int outputSize;
    functions::Activation activationFunction;
    std::vector<std::vector<double>> weightsMatrix;
    std::vector<double> biases;
};

#endif  // NEURALNETWORK_LAYER_H
