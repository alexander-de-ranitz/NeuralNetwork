//
// Created by alexander on 19-08-22.
//

#ifndef NEURALNETWORK_EVOLUTIONSIM_H
#define NEURALNETWORK_EVOLUTIONSIM_H

#include <functional>

#include "../NeuralNetwork/NeuralNetwork.h"

struct Organism {
    NeuralNetwork neuralNetwork;
    double fitness = 0.0;
};

typedef std::function<std::vector<double>(std::vector<double>)> Transformation;
class EvolutionSim {
   public:
    EvolutionSim(std::vector<int> networkShape_, Transformation targetFunction_);
    void runEvolution(int generation, int populationSize);
    std::vector<double> getCostHistory();

   private:
    static std::vector<Organism> generateNewPopulation(const std::vector<Organism>& oldPopulation);
    void calculateFitness(std::vector<Organism>& pop);
    static void sort(std::vector<Organism>& pop);
    static void storeOutput(const NeuralNetwork& neuralNetwork);

    const std::vector<int> networkShape;
    const Transformation targetFunction;
    std::vector<double> costHistory;
};

#endif  // NEURALNETWORK_EVOLUTIONSIM_H
