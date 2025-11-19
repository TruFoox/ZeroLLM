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

            // clean stdin
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            // get prompt
            std::string input;
            std::cout << "Input a message to the model:\n";
            std::getline(std::cin, input);

            // get number of tokens to generate
            std::cout << "How many tokens (characters, including spaces) should the model predict?\n";
            int tokenCount;
            while (!(std::cin >> tokenCount)) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Enter a valid number:\n";
            }
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            // Load dictionary
            std::unordered_map<std::string, int> dictionary = t.read_dict();
            if (dictionary.empty())
                throw std::runtime_error("Dictionary is empty — build it first from training data!");

            int vocab_size = (int)dictionary.size();

            // Load embeddings + weights
            std::vector<std::vector<float>> embeddings = read2DVector("../embeddings.txt", embedding_dim);
            std::vector<std::vector<std::vector<float>>> weights = read3DVector("../weights.txt", embedding_dim);

            if (embeddings.empty() || weights.empty())
                throw std::runtime_error("No existing embeddings or weights found.");

            if (weights.size() < 4) {
                weights.resize(4);
                weights[3] = std::vector<std::vector<float>>(embedding_dim, std::vector<float>(vocab_size, 0.0f));
            }

            std::vector<int> tokenSequence;
            int unkToken = dictionary["<unk>"];
            for (char c : input) {
                std::string s(1, c);
                if (dictionary.count(s))
                    tokenSequence.push_back(dictionary[s]);
                else
                    tokenSequence.push_back(unkToken); // temporary placeholder
            }

            std::vector<std::string> invDict(vocab_size);
            for (auto& p : dictionary)
                invDict[p.second] = p.first;

            for (int step = 0; step < tokenCount; step++) {

                int seqLen = tokenSequence.size();

                std::vector<std::vector<float>> vectorSeq(seqLen);
                for (int i = 0; i < seqLen; i++)
                    vectorSeq[i] = embeddings[tokenSequence[i]];

                std::vector<std::vector<float>> Q = matMul(vectorSeq, weights[0]);
                std::vector<std::vector<float>> K = matMul(vectorSeq, weights[1]);
                std::vector<std::vector<float>> V = matMul(vectorSeq, weights[2]);

                std::vector<std::vector<float>> scores = matMul(Q, transpose(K));
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        scores[i][j] /= sqrt(embedding_dim);

                for (int i = 0; i < seqLen; i++)
                    for (int j = i + 1; j < seqLen; j++)
                        scores[i][j] = -1e9f;

                std::vector<std::vector<float>> att(seqLen);
                for (int i = 0; i < seqLen; i++)
                    att[i] = softmax(scores[i]);

                std::vector<std::vector<float>> context = matMul(att, V);

                std::vector<std::vector<float>> hidden = matAdd(context, vectorSeq);
                std::vector<std::vector<float>> logits = matMul(hidden, weights[3]);

                std::vector<float> probs = softmax(logits.back());

                // Force <unk> to zero probability
                probs[unkToken] = 0.0f;

                float sumProb = 0.0f;
                for (float p : probs) sumProb += p;
                for (float& p : probs) p /= sumProb;

                int nextToken = 0;
                float best = probs[0];
                for (int i = 1; i < vocab_size; i++)
                    if (probs[i] > best) {
                        best = probs[i];
                        nextToken = i;
                    }

                // append + print
                tokenSequence.push_back(nextToken);
                std::cout << invDict[nextToken];
            }

            std::cout << "\n";
        }


	}

	return 0;
}