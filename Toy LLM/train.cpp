#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include "IO.h"
#include "train.h"
#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <mutex>
#include <algorithm>
#include <random>

using json = nlohmann::json;

/* There are a lot of comments because this is a personal learning project */

void training::buildWeights() {
    int embedding_dim = 768; // Number of floats per token embedding

    // Load dictionary: token, ID
    std::unordered_map<std::string, int> dictionary = read_dict();

    // Generate embeddings from dictionary
	std::cout << "Generating embeddings...\n";
    std::vector<std::vector<float>> updatedEmbeddings = generateEmbeddings(embedding_dim, dictionary);

    // Generate positional encodings
	std::cout << "Generating positional encodings...\n";
    std::vector<std::vector<float>> updatedPE = generatePE(embedding_dim, updatedEmbeddings);
    

    // Merge embeddings and positional encodings
	std::cout << "Combining embeddings and positional encodings...\n";
	std::vector<std::vector<float>> finalEmbeddings(updatedEmbeddings.size(), std::vector<float>(embedding_dim, 0.0f));
    
    for (size_t i = 0; i < updatedEmbeddings.size(); ++i) { // Combine them
        for (int j = 0; j < embedding_dim; ++j) {
            finalEmbeddings[i][j] = updatedEmbeddings[i][j] + updatedPE[i][j];
        }
    }

    // Save updated embeddings
    write2DVector("../embeddings.txt", finalEmbeddings);

    std::cout << "Embeddings updated. Total tokens: " << updatedEmbeddings.size() << "\n";

	// Generate weight matrices
    std::cout << "Generating weight matrices...\n";
    std::vector<std::vector<std::vector<float>>> weights = generateWeights(embedding_dim);

    // Save weight matrices
    write3DVector("../weights.txt", weights);

    std::ifstream file("../training_data.txt");
    if (!file.is_open()) throw std::runtime_error("Error opening file");

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Get # of sequences
    std::vector<int> sequence = makeSequence(data, dictionary);
    int totalSequences = (sequence.size() + 127) / 128; // ceil division

    for (int i = 0; i < totalSequences; i++) {
        int start = i * 128;
        int end = std::min(start + 128, (int)sequence.size());
        std::vector<int> currentSequence(sequence.begin() + start, sequence.begin() + end);
    }


}


// Generate embeddings aligned with dictionary (Creates dictionary of words in training data for later)
std::vector<std::vector<float>> training::generateEmbeddings(const int embedding_dim, const std::unordered_map<std::string, int>& dictionary) {

    std::vector<std::vector<float>> embeddings(embedding_dim, std::vector<float>(embedding_dim, 0.0f));

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
std::vector<std::vector<std::vector<float>>> training::generateWeights(const int embedding_dim) {
    // Random generator for initializing new embeddings
    std::random_device rd;
    std::mt19937 gen(rd());
     
    // Default value range for embeddings
     // Default range: [-1/sqrt(embedding_dim), 1/sqrt(embedding_dim)] (Reason: It sounds about right, & it scales with larger dimensions)
    std::uniform_real_distribution<float> dis((-1.0f / sqrt(embedding_dim)), 1.0f / sqrt(embedding_dim));

    // Allocate 3D vector: 4 matrices x embedding_dim x embedding_dim
    std::vector<std::vector<std::vector<float>>> weights(4, std::vector<std::vector<float>>(embedding_dim, std::vector<float>(embedding_dim)));

    /* Iterate through all weight matrices (4 per embedding dimension: Q,K,V,Output) & add random values */
	for (int i = 0; i < 4; ++i) { // For each weight matrix type:

		for (int j = 0; j < embedding_dim; ++j) { // Fill each value in matrix with random float
            for (int k = 0; k < embedding_dim; ++k) {
                weights[i][j][k] = dis(gen);
            }
        }
    }
	return weights;
}

// Splits data into sequences of sequenceLength length
std::vector<int> training::makeSequence(const std::string& data, const std::unordered_map<std::string, int>& dictionary) { // Tokenize text into token IDs
    std::string delims = " ,!?-()'.\"[];:/–\n——&{}";
    std::string currentWord;
    std::vector<int> tokenizedString;

	std::string dataLower = data; // Cant do const on a non mutable string

    for (char& c : dataLower) { // Convert to lowercase
        if (c >= 'A' && c <= 'Z') c |= 0x20;
    }

    /* Tokenization */
    for (size_t i = 0; i < dataLower.size(); ++i) {
        char currentChar = dataLower[i];

        bool isDelimiter = delims.find(currentChar) != std::string::npos;

        // Special handling for apostrophes
        if (currentChar == '\'') {
            bool prevIsLetter = (i > 0 && std::isalpha(dataLower[i - 1]));
            bool nextIsLetter = (i + 1 < dataLower.size() && std::isalpha(dataLower[i + 1]));

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
                tokenizedString.push_back(encode(currentWord, dictionary));
                currentWord.clear();
            }

            // Include delimiter itself as token (except for whitespace)
            std::string delimStr(1, currentChar);
            tokenizedString.push_back(encode(delimStr, dictionary));
        }
        else {
            currentWord += currentChar;
        }
    }

    if (!currentWord.empty()) {
        tokenizedString.push_back(encode(currentWord, dictionary));
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
    for (size_t i = 0; i < normalizedData.size(); ++i) {
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
    return std::accumulate(result.begin(), result.end(), std::string(),
        [](const std::string& a, const std::string& b) {
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
