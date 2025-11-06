#include "train.h"
#include "normalizer.h"
#include <iostream> 

using namespace std; // Not good practice but bite me lol


int main() {
	int choice;

	cout << "Please input what you want to do:\n1. Train Model\n2. Chat with Model\n";
	cin >> choice;

	if (cin.fail() || (choice != 1 && choice != 2)) { // Error handling for invalid input
		cerr << "Input error!" << endl;
		return 1;
	}

	if (choice == 1) {
		cout << "Training model..." << endl;

		training t;
		t.train();

	} else if (choice == 2) {
		// Call chat function
		cout << "Chatting with model..." << endl;
		// Placeholder for chat code
	}

	return 0;
}