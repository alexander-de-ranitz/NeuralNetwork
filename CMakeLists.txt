cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(NN
        main.cpp
        Src/Network/NeuralNetwork.cpp
        Include/NeuralNetwork/NeuralNetwork.h
        Src/Network/Layer.cpp
        Include/NeuralNetwork/Layer.h
        Src/Utilities/Random.cpp
        Include/Utilities/Random.h
        Src/Evolutionary/EvolutionSim.cpp
        Include/Evolutionary/EvolutionSim.h
        Include/Utilities/Functions.h
        Include/Utilities/Printer.h
        Src/Utilities/Functions.cpp)
