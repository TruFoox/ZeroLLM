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
    std::vector<std::vector<float>> generateEmbeddings(const int embedding_dim, const std::unordered_map<std::string, int>& dictionary);

    std::vector<float> stableSoftmax(const std::vector<float>& x);

    void normalizeVector(std::vector<float>& v);

    /* Generate positional encodings for the dictionary */
    std::vector<std::vector<float>> generatePE(int max_seq_len, int embedding_dim);

	/* Generate feedforward weights */
    std::vector<std::vector<std::vector<float>>> generateFFWeights(const int embedding_dim);

    /* Generate weight matrices */
    std::vector<std::vector<std::vector<float>>> generateWeights(const int embedding_dim, const int vocab_size);

    /* Splits into sequences */
    std::vector<int> makeSequence(const std::string& data, const std::unordered_map<std::string, int>& dictionary);

    void layerNorm(std::vector<std::vector<float>>& x, float eps = 1e-5f);

    void layerNormBackward(const std::vector<std::vector<float>>& y, const std::vector<std::vector<float>>& dy, std::vector<std::vector<float>>& dx);

    void clip(std::vector<std::vector<float>>& grad, float threshold);

    /* Write new word to dictionary */
    void define(const std::string& text, std::unordered_map<std::string, int>& dictionary);
    void loop(int embedding_dim, int intput, int threadnum, std::unordered_map<std::string, int> dictionary, std::vector<std::vector<float>> finalEmbeddings, std::vector<std::vector<std::vector<float>>> weights);

    /* Decode tokens back to text */
    std::string decode(const std::vector<int>& tokens, const std::unordered_map<std::string, int>& dictionary);
    
    /* Encode text to tokens */
    int encode(const std::string& text, const std::unordered_map<std::string, int>& dictionary);

    /* Dictionary read/write */
    void write_dict(const std::unordered_map<std::string, int>& dict);
    std::unordered_map<std::string, int> read_dict();
};
