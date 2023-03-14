//
// Created by alexander on 19-08-22.
//

#ifndef NEURALNETWORK_EVOLUTIONSIM_H
#define NEURALNETWORK_EVOLUTIONSIM_H

#include "../NeuralNetwork/NeuralNetwork.h"

struct Organism {
    NeuralNetwork neuralNetwork;
    double fitness = 0.0;
};

class EvolutionSim {
   public:
    EvolutionSim(std::vector<int> networkShape_);
    void runEvolution(int generation, int populationSize, std::vector<Data>& trainingData);
    std::vector<double> getCostHistory();

   private:
    static std::vector<Organism> generateNewPopulation(const std::vector<Organism>& oldPopulation);
    void calculateFitness(std::vector<Organism>& pop, std::vector<Data>& data);
    static void sort(std::vector<Organism>& pop);

    const std::vector<int> networkShape;
    std::vector<double> costHistory;
};

#endif  // NEURALNETWORK_EVOLUTIONSIM_H
