#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "train.h"
#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <mutex>

using json = nlohmann::json;

/* Main training function */
void training::train() {
    std::unordered_map<std::string, int> dictionary = read_dict(); // Load existing dictionary

    const size_t BUFFER_SIZE = 8192;
    std::ifstream file("../training_data.txt", std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Error opening file");

    std::vector<char> buffer(BUFFER_SIZE);
    int tokenCount = 0;
    std::string leftover; // for split tokens across buffers
    std::string delims = "_ ,!?-().";


    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
        std::streamsize bytesRead = file.gcount();
        std::string data = leftover + std::string(buffer.begin(), buffer.begin() + bytesRead);
        leftover.clear();
        std::string current;

        for (char c : data) {
            if (delims.find(c) != std::string::npos) {
                // Add word to dictionary if not already within
                if (!current.empty()) {
                    define(current, dictionary);
                    current.clear();
                }
                
				// Define delimiter as a token
                std::string delimToken(1, c);
                define(delimToken, dictionary);  // still using define()
            }
            else {
                current += c;
            }
        }

        leftover = current;  // carry over partial token
    }

    if (!leftover.empty()) {
        define(leftover, dictionary); 
        tokenCount++;
    }


    std::cout << "Total tokens: " << tokenCount << std::endl;
}



void training::define(const std::string& text, std::unordered_map<std::string, int>& dictionary) {
    if (text.empty() || text.find('\0') != std::string::npos) return; // skip nulls

	if (dictionary.find(text) == dictionary.end()) { // If token not in dictionary
		dictionary[text] = dictionary.size(); // Get next available value

        write_dict(dictionary);

        std::cout << "Added token: " << text << " to dictionary\n";
    }
    else {
        std::cout << "Token: " << text << " already in dictionary\n";
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
        j[pair.first] = pair.second; // pair.first = key, pair.second = value
    }
    std::ofstream out("../dictionary.json");
    out << j.dump(4);
}

std::unordered_map<std::string, int> training::read_dict() {
    std::ifstream in("../dictionary.json");
    nlohmann::json j;
    in >> j;
    return j.get<std::unordered_map<std::string, int>>();
}