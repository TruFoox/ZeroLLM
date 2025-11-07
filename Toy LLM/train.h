#pragma once
#include <unordered_map>
#include <string>
#include <vector>

class training {
private:
    std::unordered_map<std::string, int> vocab;

public:
    /* Build dictionary based on txt file */
    void buildDictionary();

    /* Build weights based on dictionary and dataset */
    void buildWeights();

    /* Generate embeddings for the dictionary */
    std::vector<std::vector<float>> generateEmbeddings(int embedding_dim, const std::unordered_map<std::string, int>& dictionary);

    /* Generate positional encodings for the dictionary */
    std::vector<std::vector<float>> generatePE(const int embedding_dim, std::vector<std::vector<float>> updatedEmbeddings);

    /* Encode text to tokens */
   void define(const std::string& text, std::unordered_map<std::string, int>& dictionary);

    /* Decode tokens back to text */
    std::string decode(const std::vector<int>& tokens, const std::unordered_map<std::string, int>& dictionary);

    /* Dictionary read/write */
    void write_dict(const std::unordered_map<std::string, int>& dict);
    std::unordered_map<std::string, int> read_dict();
};
