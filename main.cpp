#include "Include/Evolutionary/EvolutionSim.h"
#include "Include/NeuralNetwork/NeuralNetwork.h"
#include "Include/Utilities/Printer.h"

std::vector<double> f(std::vector<double> x) { return {x[0] * x[0]}; }

int main() {
    auto evolutionSim = EvolutionSim({1, 10, 10, 1}, f);
    evolutionSim.runEvolution(100, 500);
    auto hist = evolutionSim.getCostHistory();
    print(hist);
    return 0;
}