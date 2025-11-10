#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include "IO.h"
#include "doMath.h"
#include "train.h"
#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <mutex>
#include <numeric>
#include <algorithm>
#include <random>

using json = nlohmann::json;

/* There are a lot of comments because this is a personal learning project */
void training::buildWeights() {
    int embedding_dim = 768;
    float learning_rate = 0.001f;

    char input;
    int intput; // Stores user inputs

	std::cout << "Build new model weights? (y/n): ";
    std::cin >> input;

    // Load dictionary
    std::unordered_map<std::string, int> dictionary = read_dict();

    if (dictionary.empty())
        throw std::runtime_error("Dictionary is empty — build it first from training data!");
   
    int vocab_size = static_cast<int>(dictionary.size());


    std::vector<std::vector<float>> finalEmbeddings;
    std::vector<std::vector<std::vector<float>>> weights;

    if (input == 'y' || input == 'Y') {
        // Generate embeddings and positional encodings
        std::cout << "Generating embeddings...\n";
        std::vector<std::vector<float>> updatedEmbeddings = generateEmbeddings(embedding_dim, dictionary);

        std::cout << "Generating positional encodings...\n";
        std::vector<std::vector<float>> updatedPE = generatePE(embedding_dim, updatedEmbeddings);

        // Combine embeddings and PEs
        std::cout << "Combining embeddings and positional encodings...\n";
        finalEmbeddings.resize(updatedEmbeddings.size(), std::vector<float>(embedding_dim, 0.0f));
        for (int i = 0; i < updatedEmbeddings.size(); ++i)
            for (int j = 0; j < embedding_dim; ++j)
                finalEmbeddings[i][j] = updatedEmbeddings[i][j] + updatedPE[i][j];

        write2DVector("../embeddings.txt", finalEmbeddings);
        std::cout << "Embeddings updated. Total tokens: " << finalEmbeddings.size() << "\n";

        // Generate weight matrices (Q, K, V, WO)
        std::cout << "Generating weight matrices...\n";
        weights = generateWeights(embedding_dim, vocab_size);
        write3DVector("../weights.txt", weights);

        intput = 0;
    }
    else {
        // Load existing embeddings and weights
        std::cout << "Loading existing embeddings and weights...\n";
        finalEmbeddings = read2DVector("../embeddings.txt", embedding_dim);
        weights = read3DVector("../weights.txt", embedding_dim);

        if (finalEmbeddings.empty() || weights.empty())
            throw std::runtime_error("No existing embeddings or weights found. Please build first!");

        // Ensure correct shapes
        if (weights.size() < 4) {
            std::cout << "Fixing incomplete weights...\n";
            weights.resize(4, std::vector<std::vector<float>>(embedding_dim, std::vector<float>(embedding_dim, 0.0f)));
            weights[3] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(vocab_size, 0.0f));
        }
        else {
            for (size_t w = 0; w < weights.size(); ++w) {
                if (weights[w].empty()) continue;

                // Fix Q, K, V
                if (w < 3 && weights[w][0].size() != embedding_dim)
                    weights[w] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(embedding_dim, 0.0f));

                // Fix WO
                else if (w == 3 && weights[w][0].size() != vocab_size)
                    weights[w] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(vocab_size, 0.0f));
            }
        }

        std::cout << "Input previous sequence number: ";
        std::cin >> intput;
    }

    /* Start training weights */
    std::vector<std::vector<float>> gradWQ(embedding_dim, std::vector<float>(embedding_dim));
    std::vector<std::vector<float>> gradWK(embedding_dim, std::vector<float>(embedding_dim));
    std::vector<std::vector<float>> gradWV(embedding_dim, std::vector<float>(embedding_dim));

    std::cout << "Training model...\n";
    std::ifstream file("../training_data.txt");
    if (!file.is_open()) throw std::runtime_error("Error opening file");

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<int> sequence = makeSequence(data, dictionary);
    int totalSequences = (sequence.size() + 127) / 128; // ceil division

    // Declare context/output once
    std::vector<std::vector<float>> context;
    std::vector<std::vector<float>> output;


    // Training loop over sequences of length 128
    for (int i = intput; i < totalSequences; i++) {
        std::cout << "Training sequence #" << (i + 1) << "/" << totalSequences << "...\n";

        int start = i * 128;
        int end = std::min(start + 128, (int)sequence.size());
        int sequenceLength = end - start;
        if (sequenceLength <= 0) continue;

        // Get token embeddings for this sequence
        std::vector<int> tokenSequence(sequence.begin() + start, sequence.begin() + end);
        std::vector<std::vector<float>> vectorSequence;
        vectorSequence.reserve(sequenceLength);
        for (int token : tokenSequence) {
            if (token < 0 || token >= finalEmbeddings.size())
                throw std::runtime_error("Invalid token index");
            vectorSequence.push_back(finalEmbeddings[token]);
        }

        /* Forward pass */
        std::vector<std::vector<float>> Q = matMul(vectorSequence, weights[0]); // seq_len x embed_dim
        std::vector<std::vector<float>> K = matMul(vectorSequence, weights[1]); // seq_len x embed_dim
        std::vector<std::vector<float>> V = matMul(vectorSequence, weights[2]); // seq_len x embed_dim

        // Attention scores: seq_len x seq_len
        std::vector<std::vector<float>> attentionScores = matMul(Q, transpose(K));

        // Scale by sqrt(d)
        for (int m = 0; m < sequenceLength; ++m)
            for (int n = 0; n < sequenceLength; ++n)
                attentionScores[m][n] /= sqrt(embedding_dim);

        // Mask future tokens
        for (int m = 0; m < sequenceLength; ++m)
            for (int n = m + 1; n < sequenceLength; ++n)
                attentionScores[m][n] = -1e9;

        // Softmax
        std::vector<std::vector<float>> attentionWeights(sequenceLength, std::vector<float>(sequenceLength));
        for (int m = 0; m < sequenceLength; ++m)
            attentionWeights[m] = softmax(attentionScores[m]);

        // Context vector
        context = matMul(attentionWeights, V);

        // Output logits
        output = matMul(context, weights[3]);

        /* Backward pass */

        // Targets & Output Error
        std::vector<std::vector<float>> targets(sequenceLength, std::vector<float>(vocab_size, 0.0f));
        for (int m = 0; m < sequenceLength; ++m)
            targets[m][tokenSequence[m]] = 1.0f;

        std::vector<std::vector<float>> error(sequenceLength, std::vector<float>(vocab_size, 0.0f));
        for (int m = 0; m < sequenceLength; ++m)
            for (int n = 0; n < vocab_size; ++n)
                error[m][n] = output[m][n] - targets[m][n];

        std::vector<std::vector<float>> gradW3 = matMul(transpose(context), error);
        training::clip(gradW3, 1.0f);

        // Update WO weights
        for (int m = 0; m < embedding_dim; ++m)
            for (int n = 0; n < vocab_size; ++n)
                weights[3][m][n] -= learning_rate * gradW3[m][n];

        std::vector<std::vector<float>> dContext = matMul(error, transpose(weights[3])); // sequenceLength x embedding_dim

        gradWV = matMul(transpose(attentionWeights), dContext); // embedding_dim x embedding_dim
        training::clip(gradWV, 1.0f);

        std::vector<std::vector<float>> dAttention(sequenceLength, std::vector<float>(sequenceLength, 0.0f));
        for (int i = 0; i < sequenceLength; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < sequenceLength; ++j)
                sum += attentionWeights[i][j] * dContext[i][j];

            for (int j = 0; j < sequenceLength; ++j)
                dAttention[i][j] = attentionWeights[i][j] * (dContext[i][j] - sum);
        }

        // Gradients
        std::vector<std::vector<float>> dQ = matMul(dAttention, K); // sequenceLength x embedding_dim
        std::vector<std::vector<float>> dK = matMul(transpose(dAttention), Q); // sequenceLength x embedding_dim

        gradWQ = matMul(transpose(vectorSequence), dQ); // embedding_dim x embedding_dim
        gradWK = matMul(transpose(vectorSequence), dK); // embedding_dim x embedding_dim
        gradWV = matMul(transpose(vectorSequence), dContext); // embedding_dim x embedding_dim (fixed)

        // Clip gradients
        training::clip(gradWQ, 1.0f);
        training::clip(gradWK, 1.0f);
        training::clip(gradWV, 1.0f);

        // Update Q/K/V weights
        int embed = embedding_dim;
        for (int m = 0; m < embed; ++m) {
            for (int n = 0; n < embed; ++n) {
                weights[0][m][n] -= learning_rate * gradWQ[m][n];
                weights[1][m][n] -= learning_rate * gradWK[m][n];
                weights[2][m][n] -= learning_rate * gradWV[m][n];
            }
        }

        /* Calculate gradient for embeddings */
        std::vector<std::vector<float>> gradEmbeddings(sequenceLength, std::vector<float>(embedding_dim, 0.0f));

        // Stores contribution from Q/K/V
        std::vector<std::vector<float>> temp;

        // Q contribution
        temp = matMul(dQ, transpose(weights[0]));
        for (int t = 0; t < sequenceLength; ++t)
            for (int d = 0; d < embedding_dim; ++d)
                gradEmbeddings[t][d] += temp[t][d];

        // K contribution
        temp = matMul(dK, transpose(weights[1]));
        for (int t = 0; t < sequenceLength; ++t)
            for (int d = 0; d < embedding_dim; ++d)
                gradEmbeddings[t][d] += temp[t][d];

        // V contribution
        temp = dContext;
        for (int t = 0; t < sequenceLength; ++t)
            for (int d = 0; d < embedding_dim; ++d)
                gradEmbeddings[t][d] += temp[t][d];

        // Clip gradient
        training::clip(gradEmbeddings, 1.0f);

        // Update embeddings (subtract with learning rate)
        for (int t = 0; t < sequenceLength; ++t)
            for (int d = 0; d < embedding_dim; ++d)
                finalEmbeddings[tokenSequence[t]][d] -= learning_rate * gradEmbeddings[t][d];

        // Save updated weights
        if (i % 3 == 0)
            std::cout << "Writing to file. DO NOT QUIT\r";
            write3DVector("../weights.txt", weights);
            write2DVector("../embeddings.txt", finalEmbeddings);

    }
}


// Generate embeddings aligned with dictionary (Creates dictionary of words in training data for later)
std::vector<std::vector<float>> training::generateEmbeddings(const int embedding_dim, const std::unordered_map<std::string, int>& dictionary) {

    std::vector<std::vector<float>> embeddings(dictionary.size(), std::vector<float>(embedding_dim, 0.0f));

	std::vector<std::vector<float>> updatedEmbeddings; // New embeddings aligned with dictionary

    // Random generator for initializing new embeddings
    std::random_device rd;
    std::mt19937 gen(rd());

	// Default value range for embeddings
     // Default range: [-1/sqrt(embedding_dim), 1/sqrt(embedding_dim)] (Reason: It sounds about right, & it scales with larger dimensions)
	std::uniform_real_distribution<float> dis((-1.0f / sqrt(embedding_dim)), 1.0f / sqrt(embedding_dim));

    // Prepare new vector to store embeddings aligned with dictionary
    updatedEmbeddings.reserve(dictionary.size());

    // Iterate through dictionary
    for (const auto& [token, id] : dictionary) {

		std::vector<float> vec; // Stores embedding for current token

        // Preserve existing embedding if present
        if (id < embeddings.size() && !embeddings[id].empty()) {
            vec = embeddings[id];
        }
        else {
            // Generate new random embedding
            vec.resize(embedding_dim);
            for (auto& val : vec) {
                val = dis(gen);
            }
        }

        // Append embedding to new_weights
        updatedEmbeddings.push_back(vec);
    }

    return updatedEmbeddings;
}


// Generate positional encodings (Adds a periodic signal (cos/sin) to embeddings based on token position to distinguish index 1 from 2, etc)
std::vector<std::vector<float>> training::generatePE(const int embedding_dim, std::vector<std::vector<float>> updatedEmbeddings) {
    // Load existing encodings
    std::vector<std::vector<float>> encodings; // Holds position (Basically an index for each token in a sequence)

	// Iterate through input tokens
    for (int pos = 0; pos < updatedEmbeddings.size(); ++pos) {
        std::vector<float> vec; // Stores encodings for current token

		for (int dim = 0; dim < embedding_dim; ++dim) { // Generate 2d PE matrix

            // Calculate positional encoding (Even & odd dimension sizes require different formulas)
            if (dim % 2 == 0) {vec.push_back(sin(pos / pow(10000.0, 2.0 * floor(dim / 2.0) / embedding_dim)));}
            else {vec.push_back(cos(pos / pow(10000.0, 2.0 * floor(dim / 2.0) / embedding_dim)));};
        }

        // Add positional encodings
        encodings.push_back(vec);
    }
        
	return encodings;
}

// Generate starting value for weights (random)
std::vector<std::vector<std::vector<float>>> training::generateWeights(const int embedding_dim, const int vocab_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f / sqrt(embedding_dim), 1.0f / sqrt(embedding_dim));

    std::vector<std::vector<std::vector<float>>> weights(4);

    // Q, K, V (square matrices)
    for (int i = 0; i < 3; ++i) {
        weights[i].assign(embedding_dim, std::vector<float>(embedding_dim));
        for (int j = 0; j < embedding_dim; ++j)
            for (int k = 0; k < embedding_dim; ++k)
                weights[i][j][k] = dis(gen);
    }

    // Output layer: embedding_dim x vocab_size
    weights[3].assign(embedding_dim, std::vector<float>(vocab_size));
    for (int j = 0; j < embedding_dim; ++j)
        for (int k = 0; k < vocab_size; ++k)
            weights[3][j][k] = dis(gen);

    return weights;
}

void training::clip(std::vector<std::vector<float>>& grad, float threshold) {
    for (auto& row : grad)
        for (auto& val : row)
            val = std::max(std::min(val, threshold), -threshold);
}

// Splits data into sequences of sequenceLength length
std::vector<int> training::makeSequence(const std::string& data, const std::unordered_map<std::string, int>& dictionary) { // Tokenize text into token IDs
    std::string delims = " ,!?-()'.\"[];:/–\n——&{}";
    std::string currentWord;
    std::vector<int> tokenizedString;

    std::string dataLower = data;
    for (char& c : dataLower) {
        if (c >= 'A' && c <= 'Z') c |= 0x20; // lowercase
    }

    // Ensure dictionary has <UNK> token
    int unkId;
    auto itUnk = dictionary.find("<UNK>");
    if (itUnk != dictionary.end()) {
        unkId = itUnk->second;
    }
    else {
        throw std::runtime_error("<UNK> token missing from dictionary");
    }

    /* Tokenization */
    for (int i = 0; i < dataLower.size(); ++i) {
        char currentChar = dataLower[i];

        bool isDelimiter = delims.find(currentChar) != std::string::npos;

        // Special handling for apostrophes
        if (currentChar == '\'') {
            bool prevIsLetter = (i > 0 && std::isalpha(dataLower[i - 1]));
            bool nextIsLetter = (i + 1 < dataLower.size() && std::isalpha(dataLower[i + 1]));

            if (prevIsLetter && nextIsLetter) {
                currentWord += currentChar;
                continue;
            }
            else {
                isDelimiter = true;
            }
        }

        if (isDelimiter) {
            if (!currentWord.empty()) {
                int id = encode(currentWord, dictionary);
                tokenizedString.push_back(id >= 0 ? id : unkId); // replace -1 with <UNK>
                currentWord.clear();
            }

            if (!std::isspace(currentChar)) {
                int id = encode(std::string(1, currentChar), dictionary);
                tokenizedString.push_back(id >= 0 ? id : unkId); // replace -1 with <UNK>
            }
        }
        else {
            currentWord += currentChar;
        }
    }

    if (!currentWord.empty()) {
        int id = encode(currentWord, dictionary);
        tokenizedString.push_back(id >= 0 ? id : unkId); // replace -1 with <UNK>
    }

    return tokenizedString;
}




/* Everything below is for building the model dictionary */

void training::buildDictionary() {
    std::unordered_map<std::string, int> dictionary = read_dict(); // Load existing dictionary

    std::ifstream file("../training_data.txt");
    if (!file.is_open()) throw std::runtime_error("Error opening file");

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());


    for (char& c : data) { // Convert to lowercase
        if (c >= 'A' && c <= 'Z') c |= 0x20;
    }

    while (!data.empty() && (data.back() & 0xC0) == 0x80)
        data.pop_back();

    data.erase( // Remove byte order marker from my lazy ass not harvesting the dataset properly
        std::remove_if(data.begin(), data.end(), [](unsigned char c) {
            return c == '\xEF' || c == '\xBB' || c == '\xBF';
            }),
        data.end()
    );

    std::string normalizedData = normalize(data);

    std::string delims = " ,!?-()'.\"[];:/–\n——&{}";
    std::string currentWord;

    /* Tokenization */
    for (int i = 0; i < normalizedData.size(); ++i) {
        char currentChar = normalizedData[i];

        bool isDelimiter = delims.find(currentChar) != std::string::npos;

        // Special handling for apostrophes
        if (currentChar == '\'') {
            bool prevIsLetter = (i > 0 && std::isalpha(normalizedData[i - 1]));
            bool nextIsLetter = (i + 1 < normalizedData.size() && std::isalpha(normalizedData[i + 1]));

            if (prevIsLetter && nextIsLetter) {
                // Internal apostrophe - part of the word
                currentWord += currentChar;
                continue;
            }
            else {
                // Standalone apostrophe - treat as delimiter
                isDelimiter = true;
            }
        }

        if (isDelimiter) {
            if (!currentWord.empty()) {
                define(currentWord, dictionary);
                currentWord.clear();
            }
            define(std::string(1, currentChar), dictionary);
        }
        else {
            currentWord += currentChar;
        }

        if (i % 100 == 0) write_dict(dictionary);
    }

    if (!currentWord.empty()) define(currentWord, dictionary);


    // Add last word if any
    if (!currentWord.empty()) {
        define(currentWord, dictionary);
    }



    if (!currentWord.empty()) { // If there is a word that was cut off by the buffer size, add it
        define(currentWord, dictionary);
    }

    write_dict(dictionary);

    std::cout << "Total tokens: " << dictionary.size() << std::endl;
}




void training::define(const std::string& text, std::unordered_map<std::string, int>& dictionary) {
    // Define new token in dictionary
    if (text.empty() || text.find('\0') != std::string::npos) return; // skip nulls

    if (dictionary.find(text) == dictionary.end()) { // If token not in dictionary
        dictionary[text] = dictionary.size(); // Get next available value

        std::cout << "Token: \"" << text << "\" added to dictionary\n";
    }
    else if (text != " ") {
        // Token already exists, do nothing
        // std::cout << "Token: \"" << text << "\" already in dictionary\n";
    }
}



std::string training::decode(const std::vector<int>& tokens, const std::unordered_map<std::string, int>& dictionary) { // Decode tokens to text

	std::vector<std::string> result; // vector to hold final string

    for (const int token : tokens) {
        for (const auto& pair : vocab) { // pair.first = key, pair.second = value
            if (pair.second == token) {
				result.push_back(pair.first);
            }
        }
	}
    return std::accumulate(result.begin(), result.end(), std::string(),[](const std::string& a, const std::string& b) {
            return a + (a.length() > 0 ? " " : "") + b;
		});
}


int training::encode(const std::string& word, const std::unordered_map<std::string, int>& dictionary) {
    auto it = dictionary.find(word);
    if (it != dictionary.end()) {

        return it->second; // found: return token ID
    }
    else {
        return -1; // or some special value for unknown word
    }
}




/* Dictionary read/writing */
void training::write_dict(const std::unordered_map<std::string, int>& dict) {
    nlohmann::json j;

    for (const auto& pair : dict) {
        // Remove invalid UTF-8 characters by replacing them with '?'
        std::string clean_key;
        for (unsigned char c : pair.first) {
            if (c < 0x80) {
                clean_key += c; // ASCII stays
            }
            else {
                clean_key += '??'; // replace non-ASCII / invalid bytes
            }
        }

        j[clean_key] = pair.second;
    }

    std::ofstream out("../dictionary.json");
    // disable ensure_ascii to avoid escaping problems
    out << j.dump(4, ' ', false);
}


std::unordered_map<std::string, int> training::read_dict() {
    std::ifstream in("../dictionary.json");
    nlohmann::json j;

    // Check if file opened successfully
    if (!in.is_open()) {return {};}

    try {in >> j;} catch (...) {return {};}

    // Return empty map if JSON is null, not an object, or empty
    if (j.is_null() || !j.is_object() || j.empty()) {return {};}

    return j.get<std::unordered_map<std::string, int>>();
}
