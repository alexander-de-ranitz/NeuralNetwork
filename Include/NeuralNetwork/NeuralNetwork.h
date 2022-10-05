//
// Created by alexander on 19-08-22.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <vector>

#include "Layer.h"

class NeuralNetwork {
   public:
    NeuralNetwork() = default;

    explicit NeuralNetwork(const std::vector<int>& widths);

    void addLayers(std::vector<int> widths);

    [[nodiscard]] std::vector<double> getOutput(std::vector<double> input) const;

    [[nodiscard]] double calculateCost(const std::vector<double>& input, const std::vector<double>& expectedOutput) const;

    void mutate();

   private:
    std::vector<Layer> layers;
};

#endif  // NEURALNETWORK_NEURALNETWORK_H
