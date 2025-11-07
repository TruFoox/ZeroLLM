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

std::vector<std::vector<float>> read2DVector(const std::string& filename, const int embedding_dim) {
    std::ifstream in(filename);
    std::vector<std::vector<float>> vec;

    if (!in) {
        std::cerr << "Error opening file for reading: " << filename << "\n";
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

        // Pad row to embedding_dim
        if (row.size() < embedding_dim) row.resize(embedding_dim, 0.0f);

        vec.push_back(row);
    }

    return vec;
}


// Write a 3D vector to a file (separate 2D slices with a blank line)
void write3DVector(const std::string& filename, const std::vector<std::vector<std::vector<float>>>& vec3D) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening file for writing\n";
        return;
    }

    for (const auto& mat : vec3D) {
        for (const auto& row : mat) {
            for (size_t i = 0; i < row.size(); ++i) {
                out << row[i];
                if (i + 1 < row.size()) out << ' ';
            }
            out << '\n';
        }
        out << '\n'; // separate 2D slices
    }
}

// Read a 3D vector from a file, padding everything to embedding_dim
std::vector<std::vector<std::vector<float>>> read3DVector(const std::string& filename, const int embedding_dim) {
    std::ifstream in(filename);
    std::vector<std::vector<std::vector<float>>> vec3D;
    if (!in) {
        std::cerr << "Error opening file for reading: " << filename << "\n";
        return vec3D;
    }

    std::string line;
    std::vector<std::vector<float>> mat;

    while (std::getline(in, line)) {
        if (line.empty()) { // blank line = next 2D slice
            if (!mat.empty()) {
                vec3D.push_back(mat);
                mat.clear();
            }
            continue;
        }

        std::istringstream iss(line);
        std::vector<float> row;
        float val;
        while (iss >> val) row.push_back(val);

        if (row.size() < embedding_dim) row.resize(embedding_dim, 0.0f); // pad row
        mat.push_back(row);
    }

    if (!mat.empty()) vec3D.push_back(mat); // last slice

    for (auto& m : vec3D)
        if (m.size() < embedding_dim) m.resize(embedding_dim, std::vector<float>(embedding_dim, 0.0f)); // pad matrix

    return vec3D;
}
