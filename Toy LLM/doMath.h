#pragma once
#include <vector>


std::vector<std::vector<float>> matMul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    int m = A.size();
    int n = B[0].size();
    int p = B.size();
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[j][i] = mat[i][j];
    return result;
}

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    // Find max value for numerical stability
    float maxVal = *std::max_element(input.begin(), input.end());

    // Compute exponentials
    float sumExp = 0.0;
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal);
        sumExp += output[i];
    }

    // Normalize
    for (int i = 0; i < output.size(); ++i) {
        output[i] /= sumExp;
    }

    return output;
}