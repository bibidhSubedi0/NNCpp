// Auto Encoded Neural Network

#include <iostream>
#include <vector>
#include <string>
#include "Neuron.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "NN.hpp"
using std::cout, std::cin, std::endl, std::string, std::vector;

int main()
{
    vector<int> topology = {3, 2, 3};
    NN *Network = new NN(topology);

    vector<double> input = {1, 0, 1};
    Network->setCurrentInput(input);
    Network->setTarget(input);

    // Training Process

    double permissibleError = 0.1;
    int epoach = 0;

    // Network->forwardPropogation();
    // Network->setErrors();
    // Network->backPropogation();
    // Network->printToConsole();
    // Network->printWeightMatrices();
    // cout << "Error is : " << Network->getGlobalError();



    for (int i = 0; i < 1000; i++)
    {
        cout << "\n\n\nEpoch : " << epoach++ << endl;
        Network->forwardPropogation();
        Network->setErrors();
        Network->backPropogation();

        Network->printToConsole( );

        // cout << endl;
        // cout << endl;
        cout << "Error is : " << Network->getGlobalError();
        cout << "\n\n\n\n";
    }

    Network->printHistErrors();

    // do
    // {
    //     cout << "Epoach : " << epoach++ << endl;
    //     Network->forwardPropogation();
    //     Network->setErrors();
    //     Network->backPropogation();
    //     Network->printToConsole();

    //     // cout<<endl;
    //     // cout<<endl;
    //     cout << "Error is : " << Network->getGlobalError();
    //     cout << "\n\n\n\n";

    // } while (abs(Network->lastEpoachError()) >= abs(Network->getGlobalError()));

    // do
    // {
    //     cout << "Epoach : " << epoach++ << endl;
    //     Network->forwardPropogation();
    //     Network->setErrors();
    //     Network->backPropogation();
    //     Network->printToConsole();

    //     // cout<<endl;
    //     // cout<<endl;
    //     cout << "Error is : " << Network->getGlobalError();
    //     cout << "\n\n\n\n";

    // } while (abs(Network->getGlobalError()) >= permissibleError);

    // cout << "Final Network" << endl;
    // Network->printToConsole();
}
