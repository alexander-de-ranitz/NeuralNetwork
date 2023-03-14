#include "../../Include/Utilities/DataLoader.h"
#include <fstream>
#include <vector>
#include <sstream>

std::vector<Data> DataLoader::loadData(const std::string& fileName) {
    std::vector<Data> data;
    std::fstream file;
    file.open(fileName, std::ios::in);

    if(file.is_open())
    {
        std::vector<double> row;
        std::string line, word;

        while(getline(file, line))
        {
            row.clear();

            std::stringstream str(line);

            while(getline(str, word, ','))
                row.push_back(std::stod(word));

            auto inputData = std::vector<double>(row.begin() + 1, row.end());
            inputData = functions::scaleByFactor(inputData, 1/255.0);
            auto label = row.front();

            struct Data datum;
            datum.input = inputData;
            datum.output = toOneHot(static_cast<int>(label), 10);
            data.push_back(datum);
        }
    }
    return data;
}

std::vector<double> DataLoader::toOneHot(int classLabel, int numberOfClasses) {
    std::vector<double> output(numberOfClasses, 0.0);
    output.at(classLabel) = 1.0;
    return output;
}
