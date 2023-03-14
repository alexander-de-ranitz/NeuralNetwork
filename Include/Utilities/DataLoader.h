//
// Created by alexander on 13-03-23.
//

#ifndef NEURALNETWORK_DATALOADER_H
#define NEURALNETWORK_DATALOADER_H

#include <string>
#include <vector>
#include "../NeuralNetwork/NeuralNetwork.h"

typedef std::vector<double> vec;
typedef std::vector<std::vector<double>> matrix;



class DataLoader{
    public:
        static std::vector<Data> loadData(const std::string& fileName);

    private:
        static std::vector<double> toOneHot(int classLabel, int numberOfClasses);
};
#endif //NEURALNETWORK_DATALOADER_H
