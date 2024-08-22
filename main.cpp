// Auto Encoded Neural Network


#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include "Neuron.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "NN.hpp"
using std::cout, std::cin, std::endl, std::string, std::vector;

int main()
{
    vector<int> topology = {3, 4,4, 3};

    // Just keeping the lr 1.0001 instead of 1.0000, the final global error will be reduced by 88.535%
    double learningRate = 1.0001; 
    
    // creating a new pointer object of class NN
    NN *Network = new NN(topology,learningRate);

    vector<double> input = {1, 0, 0};
    vector<double> output = {0,1,1};
    Network->setCurrentInput(input);
    Network->setTarget(input);

    // Training Process

    double permissibleError = 0.1;
    int epoach = 0;


    do
    {
        cout << "Epoach : " << ++epoach << endl;
        Network->forwardPropogation();
        Network->setErrors();
        Network->backPropogation();
        Network->printToConsole();

        cout << "\nError is : " << Network->getGlobalError();
        cout << "\n========================================================="<<endl;;
    }  while(epoach < INT16_MAX);
    // }while (abs(Network->lastEpoachError()) >= abs(Network->getGlobalError()));
   


    cout << "\n========================================================="<<endl;
    cout << "\n========================================================="<<endl;

    Network->printWeightMatrices();    


}
