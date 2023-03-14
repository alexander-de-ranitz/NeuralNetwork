//
// Created by alexander on 19-08-22.
//

#include <iostream>

#include "../../Include/Evolutionary/EvolutionSim.h"
#include "../../Include/Utilities/Random.h"

EvolutionSim::EvolutionSim(std::vector<int> networkShape_, std::function<std::vector<double>(std::vector<double>)> targetFunction_)
    : networkShape(std::move(networkShape_)), targetFunction(std::move(targetFunction_)) {}

void EvolutionSim::runEvolution(int generations, int populationSize, std::vector<Data>& trainingData) {
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

    // Get a small random subset of the data for calculating fitness
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(trainingData), std::end(trainingData), rng);
    constexpr int TEST_DATA_SIZE = 50;
    auto testData = std::vector<Data>(trainingData.begin(), trainingData.begin() + TEST_DATA_SIZE);

    // Do initial fitness and sorting calculation to set up for the main evolution loop
    calculateFitness(pop, testData);
    sort(pop);

    for (int i = 0; i < generations; ++i) {
        // Each loop starts with a population sorted based on fitness. Then, a new population is created which is scored and sorted
        pop = generateNewPopulation(pop);

        std::shuffle(std::begin(trainingData), std::end(trainingData), rng);
        testData = std::vector<Data>(trainingData.begin(), trainingData.begin() + 50);

        calculateFitness(pop, testData);
        sort(pop);
        costHistory.emplace_back(pop.at(0).fitness);
        std::cout << "In generation " << i << std::endl;
        std::cout << "Current best fitness = " << pop.at(0).fitness << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    std::shuffle(std::begin(trainingData), std::end(trainingData), rng);
    testData = std::vector<Data>(trainingData.begin(), trainingData.begin() + TEST_DATA_SIZE);
    pop.at(0).neuralNetwork.storeOutput(testData);
}

std::vector<Organism> EvolutionSim::generateNewPopulation(const std::vector<Organism>& oldPopulation) {
    std::vector<Organism> newPopulation;
    newPopulation.reserve(oldPopulation.size());

    // Copy N-1 individuals based on exp distribution. Better individuals are more likely to be copied (rank based)
    for (int i = 0; i < oldPopulation.size() -1 ; ++i) {
        int randOrg = (int)(std::pow(Random::getRandDoubleUniform(0.0, 1.0), 2) * (double)oldPopulation.size());
        newPopulation.emplace_back(oldPopulation.at(randOrg));
    }

    // Mutate all networks
    for (auto& [nn, _] : newPopulation) nn.mutate();

    // Make sure the best individual is copied to prevent negative mutations to the best individual. This restores the pop to the original size
    newPopulation.emplace_back(oldPopulation.at(0));

    return newPopulation;
}

void EvolutionSim::calculateFitness(std::vector<Organism>& pop, std::vector<Data>& data) {
    for (auto& [nn, fitness] : pop) {
        fitness = 0;
        for (auto& example : data){
            auto cost = nn.calculateCost(example.input, example.output);
            fitness += cost;
        }
    }
}

void EvolutionSim::sort(std::vector<Organism>& pop) {
    std::sort(pop.begin(), pop.end(), [](const auto& m1, const auto& m2) { return m1.fitness < m2.fitness; });
}

std::vector<double> EvolutionSim::getCostHistory() { return costHistory; }

