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

    vector<double> input = {1, 1, 0};
    Network->setCurrentInput(input);
    Network->setTarget(input);

    // Training Process

    double permissibleError = 0.1;
    int epoach = 0;

    for (int i = 0; i < 200; i++)
    {
        cout << "Epoach : " << epoach++ << endl;
        Network->forwardPropogation();
        Network->printWeightMatrices();
        Network->setErrors();
        Network->backPropogation();
        Network->printToConsole( );
        Network->printWeightMatrices();

        cout << endl;
        cout << endl;
        cout << "Error is : " << Network->getGlobalError();
        cout << "\n\n\n\n";
    }
    // do
    // {
    //     cout << "Epoach : " << epoach++ << endl;
    //     Network->forwardPropogation();
    //     Network->setErrors();
    //     Network->backPropogation();
    //     Network->printToConsole();

    //     cout<<endl;
    //     cout<<endl;
    //     cout << "Error is : " << Network->getGlobalError();
    //     cout << "\n\n\n\n";

    // } while (abs(Network->getGlobalError()) >= permissibleError);

    cout << "Final Network" << endl;
    Network->printToConsole();
}
