//
// Created by alexander on 19-08-22.
//

#ifndef NEURALNETWORK_LAYER_H
#define NEURALNETWORK_LAYER_H

#include <vector>

class Layer {
   public:
    Layer(int inputSize_, int outputSize_);

    [[nodiscard]] std::vector<double> calculateOutput(const std::vector<double>& input) const;

    void mutateBiases();

    void mutateWeights();

    void mutateWeightsAndBiases();

   private:
    const int inputSize;
    const int outputSize;
    std::vector<double> biases;
    std::vector<std::vector<double>> weightsMatrix;
};

#endif  // NEURALNETWORK_LAYER_H
