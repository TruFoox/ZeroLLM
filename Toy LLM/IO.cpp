#include "IO.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>


void write2DVector(const std::string& filename, const std::vector<std::vector<float>>& vec) { // Write 2d vector to file
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening file for writing\n";
        return;
    }

    for (const auto& row : vec) {
        for (size_t i = 0; i < row.size(); ++i) {
            out << row[i];
            if (i + 1 < row.size())
                out << ' ';
        }
        out << '\n';
    }
}

std::vector<std::vector<float>> read2DVector(const std::string& filename) { // Read 2d vector from file
    std::ifstream in(filename);
    std::vector<std::vector<float>> vec;

    if (!in) {
        std::cerr << "Error opening file for reading\n";
        return vec;
    }

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        float val;
        while (iss >> val) {
            row.push_back(val);
        }
        vec.push_back(row);
    }

    return vec;
}