#pragma once
#include <vector>
#include <string>

std::vector<std::vector<float>> read2DVector(const std::string& filename);
void write2DVector(const std::string& filename, const std::vector<std::vector<float>>& vec);