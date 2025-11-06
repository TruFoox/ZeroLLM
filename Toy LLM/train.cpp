#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "train.h"
#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <mutex>
#include <algorithm>

using json = nlohmann::json;

/* Main training function */
void training::buildDictionary() {
    std::unordered_map<std::string, int> dictionary = read_dict(); // Load existing dictionary

    const size_t BUFFER_SIZE = 8192; // Only load 8kb at a time
    std::ifstream file("../training_data.txt", std::ios::binary);

    if (!file.is_open()) throw std::runtime_error("Error opening file");

    std::vector<char> buffer(BUFFER_SIZE);
    std::string leftover; // for split tokens across buffers
    std::string delims = " ,!?-().\"[];:/–\n——&";

    /* Start processing */
    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
        std::streamsize bytesRead = file.gcount();
        std::string data = leftover + std::string(buffer.begin(), buffer.begin() + bytesRead);
       


        // Then run normalize() and tokenization

        for (char& c : data) { // Convert to lowercase
            if (c >= 'A' && c <= 'Z') c |= 0x20;
        }

        while (!data.empty() && (data.back() & 0xC0) == 0x80)
            data.pop_back();

        data.erase( // Remove byte order marker from my lazy ass not harvesting the dataset properly
            std::remove_if(data.begin(), data.end(),[](unsigned char c) {
                    return c == '\xEF' || c == '\xBB' || c == '\xBF';
                }),
            data.end()
        );

        std::string normalizedData = normalize(data);

        leftover.clear();
        std::string current;


        /* Tokenization */
        for (size_t i = 0; i < normalizedData.size(); ++i) {
            char currentChar = normalizedData[i];

            if (delims.find(currentChar) != std::string::npos) {
                // Add word to dictionary if not already within
                if (!current.empty()) {
                    define(current, dictionary);
                    current.clear();
                }
                
				// Define delimiter as a token
                std::string delimToken(1, currentChar);
                define(delimToken, dictionary);
            }
            else {
                current += currentChar;
            }

            /* Edge case, odd-grammar handling (like when ' is used instead of ") */
            if (currentChar == '\'' && (normalizedData[i + 1] == ' ' || normalizedData[i - 1] == ' ')) { // If ' is right next to a space, treat as delimiter
                std::string delimToken(1, currentChar);
                define(delimToken, dictionary);
            }
        }


        leftover = current;  // carry over partial token
    }

	if (!leftover.empty()) { // If there is a word that was cut off by the buffer size, add it
        define(leftover, dictionary); 
    }

    std::cout << "Total tokens: " << dictionary.size() << std::endl;
}



void training::define(const std::string& text, std::unordered_map<std::string, int>& dictionary) { // Define new token in dictionary
    if (text.empty() || text.find('\0') != std::string::npos) return; // skip nulls

	if (dictionary.find(text) == dictionary.end()) { // If token not in dictionary
		dictionary[text] = dictionary.size(); // Get next available value

        write_dict(dictionary);

        std::cout << "Token: \"" << text << "\" added to dictionary\n";
    }
    else if (text != " ") {
        //std::cout << "Token: \"" << text << "\" already in dictionary\n";
    }
}


std::string training::decode(const std::vector<int>& tokens, std::unordered_map<std::string, int>& dictionary) { // Decode tokens to text

	std::vector<std::string> result; // vector to hold final string

    for (const int token : tokens) {
        for (const auto& pair : vocab) {
            if (pair.second == token) {
				result.push_back(pair.first); // pair.first = key, pair.second = value
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