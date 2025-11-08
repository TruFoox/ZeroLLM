#include "train.h"
#include "normalizer.h"
#include <iostream> 

using namespace std; // Not good practice but bite me lol


int main() {
	int choice;

	cout << "Please input what you want to do:\n1. Build Model Dictionary\n2. Build Model Weights\n3. Chat with Model\n";
	cin >> choice;

	if (cin.fail() || (choice != 1 && choice != 2 && choice != 3)) { // Error handling for invalid input
		cerr << "Input error!" << endl;
		return 1;
	}

	if (choice == 1) {;
		training t;

		t.buildDictionary();

	} else if (choice == 2) {
		training t;

		t.buildWeights();

	} else if (choice == 3) {
		cout << "Input a message to the model:" << endl;

		string input;

		std::cin >> std::ws; // Remove whitespace
		std::getline(std::cin, input);
		training t; 
		vector<int> i = t.makeSequence(input, t.read_dict());
		
		for (size_t j = 0; j < i.size(); ++j) {
			cout << i[j] << " ";
		}

	}


	return 0;
}