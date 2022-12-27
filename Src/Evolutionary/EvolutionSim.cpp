//
// Created by alexander on 19-08-22.
//

#include <iostream>

#include "../../Include/Evolutionary/EvolutionSim.h"
#include "../../Include/Utilities/Random.h"

EvolutionSim::EvolutionSim(std::vector<int> networkShape_, std::function<std::vector<double>(std::vector<double>)> targetFunction_)
    : networkShape(std::move(networkShape_)), targetFunction(std::move(targetFunction_)) {}

void EvolutionSim::runEvolution(int generations, int populationSize) {
    if (generations < 1 || populationSize < 1) throw std::invalid_argument("Invalid number of generations or populations size!");

    std::vector<Organism> pop;
    pop.reserve(populationSize);
    costHistory.clear();
    costHistory.reserve(generations);

    // Generate initial population
    for (int i = 0; i < populationSize; ++i) {
        Organism organism;
        organism.neuralNetwork = NeuralNetwork(networkShape, functions::Activation::LINEAR);
        pop.emplace_back(organism);
    }

    // Do initial fitness and sorting calculation to set up for the main evolution loop
    calculateFitness(pop);
    sort(pop);

    for (int i = 0; i < generations; ++i) {
        // Each loop starts with a population sorted based on fitness. Then, a new population is created which is scored and sorted
        pop = generateNewPopulation(pop);
        calculateFitness(pop);
        sort(pop);
        costHistory.emplace_back(pop.at(0).fitness);
        std::cout << "In generation " << i << std::endl;
    }
    pop.at(0).neuralNetwork.storeOutput({0,1});
}

std::vector<Organism> EvolutionSim::generateNewPopulation(const std::vector<Organism>& oldPopulation) {
    std::vector<Organism> newPopulation;
    newPopulation.reserve(oldPopulation.size());

    // Make sure the best individual is copied
    newPopulation.emplace_back(oldPopulation.at(0));

    // Copy individuals based on exp distribution. Better individuals are more likely to be copied (rank based)
    for (int i = 1; i < oldPopulation.size(); ++i) {
        int randOrg = (int)(std::pow(Random::getRandDoubleUniform(0.0, 1.0), 2) * (double)oldPopulation.size());
        newPopulation.emplace_back(oldPopulation.at(randOrg));
    }

    // Mutate all networks
    for (auto& [nn, _] : newPopulation) nn.mutate();

    return newPopulation;
}

void EvolutionSim::calculateFitness(std::vector<Organism>& pop) {
    for (auto& [nn, fitness] : pop) {
        fitness = 0;
        for (int i = 0; i < 100; ++i) {
            std::vector<double> input = Random::getRandDoubleUniform(-1, 1, networkShape.at(0));
            std::vector<double> expectedOutput = targetFunction(input);
            fitness += nn.calculateCost(input, expectedOutput);
        }
    }
}

void EvolutionSim::sort(std::vector<Organism>& pop) {
    std::sort(pop.begin(), pop.end(), [](const auto& m1, const auto& m2) { return m1.fitness < m2.fitness; });
}

std::vector<double> EvolutionSim::getCostHistory() { return costHistory; }

