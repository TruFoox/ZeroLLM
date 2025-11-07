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

	std::cout << "Generating embeddings...\n";
    // Generate embeddings from dictionary
    std::vector<std::vector<float>> updatedEmbeddings = generateEmbeddings(embedding_dim, dictionary);

    // Generate positional encodings
	std::cout << "Generating positional encodings...\n";
    std::vector<std::vector<float>> updatedPE = generatePE(embedding_dim, updatedEmbeddings);
    
    // Merge embeddings and positional encodings
	std::cout << "Combining embeddings and positional encodings...\n";
	std::vector<std::vector<float>> finalEmbeddings(updatedEmbeddings.size(), std::vector<float>(embedding_dim, 0.0f));
    
    for (size_t i = 0; i < updatedEmbeddings.size(); ++i) {
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

    std::string delims = " ,!?-().\"[];:/–\n——&{}";
    std::string currentWord;

    /* Tokenization */
    for (size_t i = 0; i < normalizedData.size(); ++i) {
        char currentChar = normalizedData[i];

        if (delims.find(currentChar) != std::string::npos) {
            // Add word to dictionary if not already within
            if (!currentWord.empty()) {
                define(currentWord, dictionary);
                currentWord.clear();
            }

            // Define delimiter as a token
            std::string delimToken(1, currentChar);
            define(delimToken, dictionary);
        }
        else {
            currentWord += currentChar;
        }

        /* Edge case, odd-grammar handling (like when ' is used instead of ") */
        if (currentChar == '\'' &&
            ((i + 1 < normalizedData.size() && normalizedData[i + 1] == ' ') ||
                (i > 0 && normalizedData[i - 1] == ' '))) { // If ' is right next to a space, treat as delimiter
            std::string delimToken(1, currentChar);
            define(delimToken, dictionary);
        }

        if (i % 100 == 0) {
            write_dict(dictionary); // Periodically save dictionary to avoid data loss
        }
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
    in >> j;
    return j.get<std::unordered_map<std::string, int>>();
}