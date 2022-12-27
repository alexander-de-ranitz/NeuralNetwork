#include "Include/Evolutionary/EvolutionSim.h"
#include "Include/NeuralNetwork/NeuralNetwork.h"
#include "Include/Utilities/Printer.h"
#include "Include/Utilities/Random.h"

/// The function to be optimised for
std::vector<double> f(std::vector<double> x) { return {3*x[0]+2}; }

/// Generate data to train/test on
std::vector<data> generateData(int size){
    std::vector<data> output;
    output.reserve(size);
    for (int i = 0; i < size; ++i) {
        data d;
        d.input = {i/static_cast<double>(size)};
        d.output = f(d.input);
        output.emplace_back(d);
    }
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(std::begin(output), std::end(output), rng);
    return output;
}

void runEvolutionSim(){
    auto evolutionSim = EvolutionSim({1, 1, 1}, f);
    evolutionSim.runEvolution(200, 500);
    auto hist = evolutionSim.getCostHistory();
    print(hist);
}

void runGradientDescent(){
    auto data = generateData(1000);

    auto nn = NeuralNetwork({1,2,2,2,1}, functions::Activation::LINEAR);
    nn.gradientDescent(data, 800, 0.05);

    nn.storeOutput({0,1});
}

int main() {
    //runGradientDescent();
    //runEvolutionSim();
}


