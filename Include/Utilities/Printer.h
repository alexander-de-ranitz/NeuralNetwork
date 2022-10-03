//
// Created by alexander on 18-08-22.
//

#ifndef NEURALNETWORK_PRINTER_H
#define NEURALNETWORK_PRINTER_H

#include <iostream>
#include <vector>

template <typename T>
void print(std::vector<T> input) {
    if (input.empty()) {
        std::cout << "{}" << std::endl;
        return;
    }

    if (std::is_arithmetic<T>::value) {
        std::cout << "{";
        for (int i = 0; i < input.size() - 1; ++i) {
            std::cout << input[i] << ", ";
        }
        std::cout << input[input.size() - 1] << "}" << std::endl;
    } else {
        std::cout << "Error, trying to print non-numeric type!" << std::endl;
    }
}

template <typename T>
void print(T input) {
    std::cout << input << std::endl;
}

#endif  // NEURALNETWORK_PRINTER_H
