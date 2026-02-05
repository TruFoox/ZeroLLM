#pragma once
#include <vector>

inline std::vector<std::vector<float>> matMul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
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

inline std::vector<std::vector<float>> matAdd(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    int m = A.size();
    int n = A[0].size();
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];

    return C;
}

inline float relu(float x) {
    return (x > 0) ? x : 0;
}

inline void relu(std::vector<std::vector<float>>& mat) {
    for (auto& row : mat)
        for (auto& val : row)
            val = relu(val); 
}

inline float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result = a[i] * b[i];
    }
    return result;
}

inline std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[j][i] = mat[i][j];
    return result;
}

inline std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    float maxVal = *std::max_element(input.begin(), input.end());
    float sumExp = 0.0;

    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxVal);
        sumExp += output[i];
    }

    for (int i = 0; i < output.size(); ++i)
        output[i] /= sumExp;

    return output;
}
