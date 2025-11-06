#pragma once
#include <unordered_map>
#include <string>
#include <vector>

class training {
    private:
        std::unordered_map<std::string, int> vocab;
    public:
		void train(); // Main training function

		void define(const std::string& text, std::unordered_map<std::string, int>& dictionary); // Encode text to tokens

		std::string decode(const std::vector<int>& tokens, std::unordered_map<std::string, int>& dictionary); // Decode tokens to text

		/* Dictionary read/writing */
        void write_dict(const std::unordered_map<std::string, int>& dict);
        std::unordered_map<std::string, int> read_dict();

};

