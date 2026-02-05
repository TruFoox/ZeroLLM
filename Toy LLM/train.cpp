#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <unordered_map>
#define NOMINMAX
#include <math.h>
#include "IO.h"
#include "doMath.h"
#include "train.h"
#include "normalizer.h"
#include <nlohmann/json.hpp>
#include <mutex>
#include <windows.h>
#include <algorithm>
#include <random>

using json = nlohmann::json;

std::mutex updateMutex; // protects shared weights/embeddings
std::mutex printMutex; // Stops jumbled console output

std::atomic<int> sequencesProcessed;
std::atomic<bool> keepTraining{ true };
std::atomic<bool> useConsole{ true };

/* There are a lot of comments because this is a personal learning project
   - This model is per-sequence SGD*/
void training::buildWeights() {
    keepTraining.store(true);
    sequencesProcessed.store(0);

    int embedding_dim = 256;
    float learning_rate = 1e-4;

    // Positional encoding:
    // - Tokens by themselves don't tell the model their position in the sentence.
    // - We add a fixed pattern (sine/cosine) to each token embedding depending on its position.
    // - This makes the same word at position 2 different from the same word at position 5.
    const std::vector<std::vector<float>> pe = generatePE(128, embedding_dim);

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
    std::vector<std::vector<std::vector<float>>> weights, FFWeights;

    if (input == 'y' || input == 'Y') {
        // Generate embeddings and positional encodings
        std::cout << "Generating embeddings...\n";
        finalEmbeddings = generateEmbeddings(embedding_dim, dictionary);

        write2DVector("../embeddings.txt", finalEmbeddings);
        std::cout << "Embeddings updated. Total tokens: " << finalEmbeddings.size() << "\n";

        // Generate weight matrices (Q, K, V, WO)
        std::cout << "Generating weight matrices...\n";
        weights = generateWeights(embedding_dim, vocab_size);
        write3DVector("../weights.txt", weights);

		std::cout << "Generating Feed-Forward Weights...\n";
		FFWeights = generateFFWeights(embedding_dim);
        write3DVector("../FFweights.txt", FFWeights);

        intput = 0;
    }
    else {
        // Load existing embeddings and weights
        std::cout << "Loading existing embeddings and weights...\n";

        finalEmbeddings = read2DVector("../embeddings.txt", embedding_dim);
        weights = read3DVector("../weights.txt", embedding_dim);
        FFWeights = read3DVector("../FFweights.txt", embedding_dim);

        if (finalEmbeddings.empty() || weights.empty())
            throw std::runtime_error("No existing embeddings or weights found. Please build first!");

        std::cout << "Input previous sequence number: ";
        std::cin >> intput;
    }

    /* Start training weights */

    std::ifstream file("../training_data.txt");
    if (!file.is_open()) throw std::runtime_error("Error opening file");

    std::string data((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    std::vector<int> sequence = makeSequence(data, dictionary);
    int totalSequences = (sequence.size() + 127) / 128; // ceil division

    // Training loop over sequences of length 128
    auto trainSubset = [&](int threadNum, int numThreads) mutable {
        std::vector<std::vector<float>> gradWQ(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        std::vector<std::vector<float>> gradWK(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        std::vector<std::vector<float>> gradWV(embedding_dim, std::vector<float>(embedding_dim, 0.0f));
        std::vector<std::vector<float>> gradW3(embedding_dim, std::vector<float>(vocab_size, 0.0f));

        // Declare context/output once
        std::vector<std::vector<float>> context;
        std::vector<std::vector<float>> output;

        for (int i = intput + threadNum; i < totalSequences; i += numThreads) {

            // Zero per-sequence gradients
            for (auto& r : gradWQ) std::fill(r.begin(), r.end(), 0.0f);
            for (auto& r : gradWK) std::fill(r.begin(), r.end(), 0.0f);
            for (auto& r : gradWV) std::fill(r.begin(), r.end(), 0.0f);
            for (auto& r : gradW3) std::fill(r.begin(), r.end(), 0.0f);

            int start = i * 128;
            int end = std::min(start + 128, (int)sequence.size());
            int sequenceLength = end - start;
            if (sequenceLength <= 0) continue;

            // Get token embeddings for this sequence
            std::vector<int> tokenSequence(sequence.begin() + start,
                sequence.begin() + end);

            std::vector<std::vector<float>> vectorSequence;
            vectorSequence.reserve(sequenceLength);
            std::vector<std::vector<float>> embeddingOnly;
            embeddingOnly.reserve(sequenceLength);


            for (int t = 0; t < tokenSequence.size(); ++t) {
                int token = tokenSequence[t];
                if (token < 0 || token >= finalEmbeddings.size())
                    throw std::runtime_error("Invalid token index");

                std::vector<float> emb = finalEmbeddings[token];
                embeddingOnly.push_back(emb);

                std::vector<float> v = emb;
				// Add positional encoding
                for (int d = 0; d < embedding_dim; ++d)
                    v[d] += pe[t][d];

                vectorSequence.push_back(v);
            }



            /* Forward pass */
            std::vector<std::vector<float>> Q, K, V;


            Q = matMul(vectorSequence, weights[0]);
            K = matMul(vectorSequence, weights[1]);
            V = matMul(vectorSequence, weights[2]);
 


            // Attention logits and scaling:
            // - We compute similarity scores between tokens: score = Query · Key.
            // - We divide those scores by sqrt(d) so they stay in a sensible numeric range.
            // - These raw scores are called "logits" (just unnormalized numbers).
            std::vector<std::vector<float>> attentionScores = matMul(Q, transpose(K));
            for (int m = 0; m < sequenceLength; ++m)
                for (int n = 0; n < sequenceLength; ++n) {
                    attentionScores[m][n] *= (1.0f / sqrtf((float)embedding_dim));

                }

            // Causal mask: set scores for future positions to a very large negative number so they don't contribute (That makes their softmax probability effectively zero.)
            for (int m = 0; m < sequenceLength; ++m)
                for (int n = m + 1; n < sequenceLength; ++n)
                    attentionScores[m][n] = -1e9f;

            // Softmax and loss:
            // - Softmax: turns a set of logits into probabilities that add to 1.
            // - Cross-entropy loss for a position = -log(probability of the true next token).
            // - We average this loss across positions. The loss tells us how bad the model's guess was.
            std::vector<std::vector<float>> attentionWeights(sequenceLength,
                std::vector<float>(sequenceLength));
            for (int m = 0; m < sequenceLength; ++m)
                attentionWeights[m] = softmax(attentionScores[m]);

            // Context
            context = matMul(attentionWeights, V);

            // Residual connection (simple):
            // - We add the attention result (context) to the original input vector.
            // - This "shortcut" helps the model keep the original token info and makes learning easier.
            std::vector<std::vector<float>> hidden = matAdd(context, vectorSequence);
            layerNorm(hidden);

            auto hidden_before_ffn = hidden;

            // Feed Forward

            std::vector<std::vector<float>> ff1 = matMul(hidden, FFWeights[0]);
            relu(ff1);
            std::vector<std::vector<float>> ff2 = matMul(ff1, FFWeights[1]);

            hidden = matAdd(hidden, ff2);
            layerNorm(hidden);

            // Final projection
            output = matMul(hidden, weights[3]);


            // Softmax on output
            std::vector<std::vector<float>> outputProb(sequenceLength,
                std::vector<float>(vocab_size));
            for (int t = 0; t < sequenceLength; ++t)
                outputProb[t] = training::stableSoftmax(output[t]);




            // Backprop through attention softmax:
            // - For each query row, softmax turns logits into weights s.
            // - Upstream gradient g for those weights turns into logits-gradient d using:
            //     d = s * (g - (s·g))
            // - This computes how much to change the attention logits so the resulting weights change in the right direction.
            std::vector<std::vector<float>> gradEmbeddings(
                sequenceLength,
                std::vector<float>(embedding_dim, 0.0f)
            );

            float combinedLoss = 0.0f;
            for (int t = 0; t + 1 < sequenceLength; ++t)
                combinedLoss += -log(outputProb[t][tokenSequence[t + 1]] + 1e-9f);

            combinedLoss /= (sequenceLength - 1);


			if (useConsole) {
                {
                    std::lock_guard<std::mutex> lock(printMutex);

                    for (int t = 0; t < sequenceLength - 1; ++t) {

                        int predictedToken = -1;
                        float bestProb = -1.0f;

                        // Calculate predicted token
                        for (int v = 0; v < vocab_size; ++v) {
                            if (v == 0) continue; // forbid UNK
                            if (outputProb[t][v] > bestProb) {
                                bestProb = outputProb[t][v];
                                predictedToken = v;
                            }
                        }

                        int actualToken = tokenSequence[t + 1];

                        std::cout
                            << "Thread " << threadNum
                            << " | Sequence #" << (i + 1) << "/" << totalSequences
                            << " | Position " << t
                            << " | Loss " << combinedLoss
                            << " | Predicted: " << decode({ predictedToken }, dictionary)
                            << " | Actual: " << decode({ actualToken }, dictionary)
                            << std::endl;
                    }
                }
            }



            /* Backward pass */

            // Backpropagation:
            // - We look at the difference between predicted probabilities and the true next token.
            // - That difference is the "error." We push that error backwards through the math below
            //   to figure out small changes to the weights that would make future predictions better.
            // - In short: measure the mistake, then nudge weights a little so next time the model errs less.

            // Compute error
            std::vector<std::vector<float>> error(sequenceLength, std::vector<float>(vocab_size, 0.0f));
            for (int m = 0; m + 1 < sequenceLength; ++m)
                for (int n = 0; n < vocab_size; ++n)
                    error[m][n] =
                    (outputProb[m][n] - (n == tokenSequence[m + 1] ? 1.0f : 0.0f));



            std::fill(error.back().begin(), error.back().end(), 0.0f);

            // Gradient clipping and accumulation:
            // - We clip gradients to a small range (e.g. [-5,5]) so a single batch doesn't blow up weights.
            // - Per-thread gradients are then added into the global accumulators under a mutex.
            // - After all threads finish we take an SGD step: weights -= learning_rate * accumulated_gradients.
            gradW3 = matMul(transpose(hidden), error);
            training::clip(gradW3, 5.0f);



            // Backprop through W_o
            std::vector<std::vector<float>> dHidden = matMul(error, transpose(weights[3]));
            for (int t = 0; t < sequenceLength; ++t)
                for (int d = 0; d < embedding_dim; ++d)
                    gradEmbeddings[t][d] += dHidden[t][d];


            // Split residual
            std::vector<std::vector<float>> dContext = dHidden; // context path
            std::vector<std::vector<float>> dHidden_ff = dHidden;  // FFN


            /* FFN Backward Pass*/

            auto gradW_ff2 = matMul(transpose(ff1), dHidden_ff);
            training::clip(gradW_ff2, 5.0f);

            // Backprop through FF2
            auto dFF1 = matMul(dHidden_ff, transpose(FFWeights[1]));

            for (int t = 0; t < sequenceLength; ++t)
                for (int d = 0; d < dFF1[0].size(); ++d)
                    if (ff1[t][d] <= 0.0f)
                        dFF1[t][d] = 0.0f;

            // Grad for FF1
            auto gradW_ff1 = matMul(transpose(hidden_before_ffn), dFF1);
            training::clip(gradW_ff1, 5.0f);

            // Backprop to hidden
            auto dHidden_from_ff = matMul(dFF1, transpose(FFWeights[0]));

            // Merge FFN gradient into dContext before attention
            for (int t = 0; t < sequenceLength; ++t)
                for (int d = 0; d < embedding_dim; ++d)
                    dContext[t][d] += dHidden_from_ff[t][d];

            // Values (V) contribution:
            // - gradWV is how much the V projection matrix should change for this sequence.
            // - The attention weights decide how much each value contributed to the context.
            // - We backprop through those weights to get gradients for V.
            // - Those gradients are also pushed back to the input vectors so token embeddings update correctly.
            auto dV = matMul(transpose(attentionWeights), dContext);
            gradWV = matMul(transpose(vectorSequence), dV);


            std::vector<std::vector<float>> dV_to_X =
                matMul(dContext, transpose(weights[2]));

            for (int t = 0; t < sequenceLength; ++t)
                for (int d = 0; d < embedding_dim; ++d)
                    gradEmbeddings[t][d] += dV_to_X[t][d];

            // Gradient for attention
            std::vector<std::vector<float>> tempAttention = matMul(dContext, transpose(V));

            // Masked positions must not affect gradients:
            // - We zero gradients corresponding to future positions because they were blocked in forward pass.
            // - This keeps learning consistent: if a forward value was ignored, its gradient is ignored too.
            for (int m = 0; m < sequenceLength; ++m)
                for (int n = m + 1; n < sequenceLength; ++n)
                    tempAttention[m][n] = 0.0f;


            std::vector<std::vector<float>> dAttention(sequenceLength, std::vector<float>(sequenceLength, 0.0f));
            for (int m = 0; m < sequenceLength; ++m) {
                float dot_sum = 0.0f;

                for (int n = 0; n <= m; ++n)
                    dot_sum += attentionWeights[m][n] * tempAttention[m][n];

                for (int n = 0; n < sequenceLength; ++n) {
                    if (n > m) {
                        dAttention[m][n] = 0.0f;   // MASK BACKPROP
                    }
                    else {
                        dAttention[m][n] =
                            attentionWeights[m][n] * (tempAttention[m][n] - dot_sum);
                    }
                }
            }


            float scale = 1.0f / sqrtf((float)embedding_dim);
            for (int m = 0; m < sequenceLength; ++m)
                for (int n = 0; n < sequenceLength; ++n)
                    dAttention[m][n] *= scale;

            std::vector<std::vector<float>> dQ = matMul(dAttention, K);
            std::vector<std::vector<float>> dK = matMul(transpose(dAttention), Q);
            


            gradWQ = matMul(transpose(vectorSequence), dQ);
            gradWK = matMul(transpose(vectorSequence), dK);
            training::clip(gradWQ, 5.0f);
            training::clip(gradWK, 5.0f);
            training::clip(gradWV, 5.0f);


            std::vector<std::vector<float>> temp;
            temp = matMul(dQ, transpose(weights[0]));
            for (int t = 0; t < sequenceLength; ++t)
                for (int d = 0; d < embedding_dim; ++d)
                    gradEmbeddings[t][d] += temp[t][d];

            temp = matMul(dK, transpose(weights[1]));
            for (int t = 0; t < sequenceLength; ++t)
                for (int d = 0; d < embedding_dim; ++d)
                    gradEmbeddings[t][d] += temp[t][d];


            training::clip(gradEmbeddings, 5.0f);


            std::unordered_map<int, std::vector<float>> tokenGradients;

            // Combine gradients for each token
            for (int t = 0; t < sequenceLength; ++t) {
                int token = tokenSequence[t];
                auto& g = tokenGradients[token];
                if (g.empty()) g.resize(embedding_dim, 0.0f);

                for (int d = 0; d < embedding_dim; ++d)
                    g[d] += gradEmbeddings[t][d];
            }

            {
                std::lock_guard<std::mutex> lock(updateMutex);

                // Q, K, V
                for (int m = 0; m < embedding_dim; ++m) {
                    for (int n = 0; n < embedding_dim; ++n) {
                        weights[0][m][n] -= learning_rate * gradWQ[m][n];
                        weights[1][m][n] -= learning_rate * gradWK[m][n];
                        weights[2][m][n] -= learning_rate * gradWV[m][n];
                    }
                }

                // FFN weights
                for (int m = 0; m < FFWeights[0].size(); ++m)
                    for (int n = 0; n < FFWeights[0][0].size(); ++n)
                        FFWeights[0][m][n] -= learning_rate * gradW_ff1[m][n];
                for (int m = 0; m < FFWeights[1].size(); ++m)
                    for (int n = 0; n < FFWeights[1][0].size(); ++n)
                        FFWeights[1][m][n] -= learning_rate * gradW_ff2[m][n];

                // Output projection
                for (int m = 0; m < embedding_dim; ++m)
                    for (int n = 0; n < vocab_size; ++n)
                        weights[3][m][n] -= learning_rate * gradW3[m][n];


                // Embeddings
                for (auto& [token, grad] : tokenGradients)
                    for (int d = 0; d < embedding_dim; ++d)
                        finalEmbeddings[token][d] -= learning_rate * grad[d];

            }



			if (GetAsyncKeyState(VK_PAUSE) & 0x8000 || !keepTraining.load(std::memory_order_relaxed)) // test for pause key
            {
                std::cout << "\nQuitting thread #" << threadNum;
                keepTraining.store(false, std::memory_order_relaxed);
                break;
            }

            if (GetAsyncKeyState(VK_HOME) & 0x8000) // test for home key (skip cout)
            {
                useConsole.store(false);
            }

            if (GetAsyncKeyState(VK_END) & 0x8000) // test for end key (show cout)
            {
                useConsole.store(true);
            }

			// Autosave every 50 sequences
            int seqCount = ++sequencesProcessed;

            if (seqCount % 50 == 0) {
                std::lock_guard<std::mutex> lock(updateMutex);
                std::cout << "\nAutosaving at sequence " << seqCount << "...\n";
                write3DVector("../weights.txt", weights);
                write2DVector("../embeddings.txt", finalEmbeddings);
                write3DVector("../FFweights.txt", FFWeights);
            }
        }

    };

	// Start training loop
    while (keepTraining.load()) {
        keepTraining = true;
        int threads;

        std::cout << "Input # of threads: ";
        std::cin >> threads;


        std::vector<std::thread> workers;
        workers.reserve(threads);

        // launch all threads
        for (int i = 0; i < threads; i++) {
            workers.emplace_back(trainSubset, i, threads);
        }

        // join all threads
        for (auto& t : workers) {
            t.join();
        }

        


        write3DVector("../weights.txt", weights);
        write2DVector("../embeddings.txt", finalEmbeddings);


        std::cout << "\nTraining complete!\n";
        keepTraining.store(false);

    }
}



// Generate embeddings aligned with dictionary (Creates dictionary of words in training data for later)
std::vector<std::vector<float>> training::generateEmbeddings(const int embedding_dim, const std::unordered_map<std::string, int>& dictionary) {

    std::vector<std::vector<float>> embeddings;

    if (std::ifstream("../embeddings.txt").good()) {
        embeddings = read2DVector("../embeddings.txt", embedding_dim);
    }


    std::vector<std::vector<float>> updatedEmbeddings; // New embeddings aligned with dictionary

    // Random generator for initializing new embeddings
    std::random_device rd;
    std::mt19937 gen(rd());

    // Default value range for embeddings
     // Default range: [-1/sqrt(embedding_dim), 1/sqrt(embedding_dim)] (Reason: It sounds about right, & it scales with larger dimensions)
    std::uniform_real_distribution<float> dis((-1.0f / sqrt(embedding_dim)), 1.0f / sqrt(embedding_dim));

    // Prepare new vector to store embeddings aligned with dictionary
    updatedEmbeddings.resize(dictionary.size());

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
        updatedEmbeddings[id] = vec; 
    }

    return updatedEmbeddings;
}



// Generate positional encodings (Adds a periodic signal (cos/sin) to embeddings based on token position to distinguish index 1 from 2, etc)
std::vector<std::vector<float>> training::generatePE(int max_seq_len, int embedding_dim) {
    std::vector<std::vector<float>> pe(max_seq_len, std::vector<float>(embedding_dim, 0.0f));

    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < embedding_dim; ++i) {
            if (i % 2 == 0)
                pe[pos][i] = sin(pos / pow(10000.0, 2.0 * (i / 2) / embedding_dim));
            else
                pe[pos][i] = cos(pos / pow(10000.0, 2.0 * (i / 2) / embedding_dim));
        }
    }

    return pe;
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

std::vector<std::vector<std::vector<float>>> training::generateFFWeights(const int embedding_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(
        -1.0f / sqrt(embedding_dim),
        1.0f / sqrt(embedding_dim)
    );

    // weights[0] = W_ff1 (d × 4d)
    // weights[1] = W_ff2 (4d × d)
    std::vector<std::vector<std::vector<float>>> weights(2);

    weights[0].resize(embedding_dim,
        std::vector<float>(embedding_dim * 4));

    for (int i = 0; i < embedding_dim; ++i)
        for (int j = 0; j < embedding_dim * 4; ++j)
            weights[0][i][j] = dis(gen);

    weights[1].resize(embedding_dim * 4,
        std::vector<float>(embedding_dim));

    for (int i = 0; i < embedding_dim * 4; ++i)
        for (int j = 0; j < embedding_dim; ++j)
            weights[1][i][j] = dis(gen);

    return weights;
}

void training::clip(std::vector<std::vector<float>>& grad, float threshold) {
    for (auto& row : grad)
        for (auto& val : row)
            val = std::max(std::min(val, threshold), -threshold);
}

std::vector<float> training::stableSoftmax(const std::vector<float>& x) {
    float maxVal = *std::max_element(x.begin(), x.end());
    std::vector<float> exps(x.size());
    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = std::exp(x[i] - maxVal);
        sum += exps[i];
    }

    for (float& v : exps) v /= sum;
    return exps;
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


            int id = encode(std::string(1, currentChar), dictionary);
            tokenizedString.push_back(id >= 0 ? id : unkId); // replace -1 with <UNK>
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
	std::unordered_map<std::string, int> timesSeen; // Tracks token frequencies

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

    // Add unknown token
    define("<UNK>", dictionary);

    /* Tokenization */
    for (int i = 0; i < normalizedData.size(); ++i) {
        char currentChar = normalizedData[i];

        bool isDelimiter = delims.find(currentChar) != std::string::npos;

        // Special handling for apostrophes used instead of "
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

		if (isDelimiter) { // If delimiter is hit, log current word & delimiter
            if (!currentWord.empty()) {
				std::cout << "Found word: " << currentWord << std::endl;
                timesSeen[currentWord]++; // Track word frequency

                currentWord.clear();
            }
            std::cout << "Found delimiter: " << std::string(1, currentChar) << std::endl;


            // Track delimiter frequency
            timesSeen[std::string(1, currentChar)]++;
        }
        else {
            currentWord += currentChar;

        }
    }

	if (!currentWord.empty()) { // Last word
        std::cout << "Found word: " << currentWord << std::endl;
        timesSeen[currentWord]++;
    }


    // Convert timesSeen to a vector of pairs (I could have done this to begin with but timesSeen[currentWord]++ is easier)
    std::vector<std::pair<std::string, int>> freqVec(timesSeen.begin(), timesSeen.end());

    // Sort by frequency descending
    std::sort(freqVec.begin(), freqVec.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });

    // Only keep top 66% of words
    int limit = static_cast<int>(freqVec.size() * 0.75);
    for (int i = 0; i < limit; ++i) {
		if (freqVec[i].second > 10) { // Remove rare words
            define(freqVec[i].first, dictionary);
        }
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



std::string training::decode(const std::vector<int>& tokens, const std::unordered_map<std::string, int>& dictionary) {
    // Build reverse map once
    std::unordered_map<int, std::string> reverseDict;
    for (const auto& pair : dictionary)
        reverseDict[pair.second] = pair.first;


    std::string result;
    result.reserve(tokens.size() * 8); // rough estimate to avoid reallocations

    for (int token : tokens) {
        auto it = reverseDict.find(token);
        if (it != reverseDict.end())
            result += it->second;
    }


    return result;
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

void training::layerNorm(std::vector<std::vector<float>>& x, float eps) {
    for (auto& row : x) {
        float mean = 0.0f;
        float var = 0.0f;

        for (float v : row) mean += v;
        mean /= row.size();

        for (float v : row) {
            float d = v - mean;
            var += d * d;
        }
        var /= row.size();

        float invStd = 1.0f / std::sqrt(var + eps);

        for (float& v : row)
            v = (v - mean) * invStd;
    }
}
