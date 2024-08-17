#include <iostream>
#include <vector>
#include <string>
#include "NeuralNetwork.hpp"
using std::cout, std::cin, std::endl, std::string, std::vector;

int main()
{
    vector<int> topology = {3, 4, 4, 3};
    vector<vector<double>> inputs = {
        {1, 0, 0}
    };

    double learningRate = 1;


    NN *Network = new NN(topology, learningRate);

    double permissibleError = 0.1;
    int epoach = 0;

    double epochError = 0.0;
    // Training Process
    while (epochError <= Network->gethisterrors().at(Network->gethisterrors().size() - 1))
    {
        epochError = 0.0;

        // Iterate over each training example
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            vector<double> input = inputs[i];
            vector<double> target = inputs[i];

            Network->setCurrentInput(input);
            Network->setTarget(target);

            Network->forwardPropogation();
            Network->setErrors();
            Network->backPropogation();

            epochError += Network->getGlobalError(); // Accumulate error for the epoch

            cout << "\n\n";
            Network->printToConsole();
            cout << "\n\n";
        }

        cout << "Epoch : " << epoach++ << endl;
        epochError = (epochError / inputs.size()); // << endl; // Average error

        // Network->printWeightMatrices();
    }

    cout << epochError << "  " << Network->gethisterrors().at(Network->gethisterrors().size() - 1) << endl;

    delete Network;
    return 0;
}
