#define NOMINMAX
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
#include <windows.h>
#include <numeric>
#include <algorithm>
#include <random> 

using namespace std; // Not good practice but bite me lol


int main() {
	while (true) {
		int choice;

		cout << "Please input what you want to do:\n1. Build Model Dictionary\n2. Build Model Weights\n3. Chat with Model\n";
		cin >> choice;

		if (cin.fail() || (choice != 1 && choice != 2 && choice != 3)) { // Error handling for invalid input
			cerr << "Input error!" << endl;
			return 1;
		}

		if (choice == 1) {
			;
			training t;

			t.buildDictionary();

		}
		else if (choice == 2) {
			training t;

			t.buildWeights();

		}
        else if (choice == 3) {
            training t;
            int embedding_dim = 256;
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            std::string input;
            std::cout << "Input a message to the model:\n";
            std::getline(std::cin, input);

            std::cout << "How many tokens (characters, including spaces) should the model predict?\n";
            int tokenCount;
            while (!(std::cin >> tokenCount)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Enter a valid number:\n";
            }
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            std::unordered_map<std::string, int> dictionary = t.read_dict();
            if (dictionary.empty())
                throw std::runtime_error("Dictionary is empty — build it first from training data!");
            int vocab_size = (int)dictionary.size();
            int unkToken = dictionary["<unk>"];

            std::unordered_map<int, std::string> invDict;
            for (auto& p : dictionary)
                invDict[p.second] = p.first;

            std::vector<std::vector<float>> embeddings = read2DVector("../embeddings.txt", embedding_dim);
            std::vector<std::vector<std::vector<float>>> weights = read3DVector("../weights.txt", embedding_dim);
            if (embeddings.empty() || weights.empty())
                throw std::runtime_error("No existing embeddings or weights found.");
            if (weights.size() < 4) {
                weights.resize(4);
                weights[3] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(vocab_size, 0.0f));
            }

            vector<int> tokenSequence = t.makeSequence(input, dictionary);
            float temperature = 0.8f; // adjust for randomness

            for (int step = 0; step < tokenCount; ++step) {
                int seqLen = tokenSequence.size();
                std::vector<std::vector<float>> vectorSeq(seqLen);

                for (int i = 0; i < seqLen; ++i) {
                    int token = tokenSequence[i];
                    if (token < 0 || token >= embeddings.size()) token = unkToken;
                    vectorSeq[i] = embeddings[token];
                }

                // Forward pass
                std::vector<std::vector<float>> Q = matMul(vectorSeq, weights[0]);
                std::vector<std::vector<float>> K = matMul(vectorSeq, weights[1]);
                std::vector<std::vector<float>> V = matMul(vectorSeq, weights[2]);

                std::vector<std::vector<float>> scores = matMul(Q, transpose(K));
                for (int i = 0; i < seqLen; ++i)
                    for (int j = 0; j < seqLen; ++j)
                        scores[i][j] /= sqrt(embedding_dim);

                // Causal mask
                for (int i = 0; i < seqLen; ++i)
                    for (int j = i + 1; j < seqLen; ++j)
                        scores[i][j] = -1e9f;

                std::vector<std::vector<float>> att(seqLen);
                for (int i = 0; i < seqLen; ++i)
                    att[i] = softmax(scores[i]);

                std::vector<std::vector<float>> context = matMul(att, V);
                std::vector<std::vector<float>> hidden = matAdd(context, vectorSeq);
                std::vector<std::vector<float>> output = matMul(hidden, weights[3]);

                // Softmax for last token
                std::vector<float> prob = softmax(output[seqLen - 1]);

                // Apply temperature
                float sum = 0.0f;
                for (int i = 0; i < prob.size(); ++i) {
                    prob[i] = pow(prob[i], 1.0f / temperature);
                    sum += prob[i];
                }
                for (auto& p : prob) p /= sum;

                // Sample from probability distribution
                float r = ((float)rand() / RAND_MAX);
                float cumulative = 0.0f;
                int predictedToken = 0;
                for (int i = 0; i < prob.size(); ++i) {
                    cumulative += prob[i];
                    if (r <= cumulative) {
                        predictedToken = i;
                        break;
                    }
                }

                tokenSequence.push_back(predictedToken);

                // Print only the new token
                std::cout << invDict[predictedToken+1];
            }
            std::cout << std::endl;

            std::cout << "\n";
        }



	}

	return 0;
}