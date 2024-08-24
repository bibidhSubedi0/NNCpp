// include guard
#ifndef _NN_HPP_
#define _NN_HPP_

#include <iostream>
#include "Matrix.hpp"
#include "Layer.hpp"
#include <vector>
using namespace std;

class NN
{
private:
    int topologySize;
    vector<int> topology;
    vector<Layer *> layers;
    vector<Matrix *> weightMatrices;
    vector<Matrix *> GradientMatrices;
    vector<double> input;
    vector<Matrix *> biasMatrices;
    double error;
    vector<double> target;
    vector<double> errors;
    vector<double> histErrors;
    double learningRate;

public:
    NN(vector<int> topology, double learningRate);
    void setCurrentInput(vector<double> input);
    void printToConsole();
    void printWeightMatrices();
    void printBiases();
    void forwardPropogation();
    void backPropogation();
    Layer *GetLayer(int nth);

    void setErrors();
    void setTarget(vector<double> target);

    void printErrors();
    double getGlobalError();
    double lastEpoachError();
    void printHistErrors();
    double getLearningRate();

    // FOR JSON
    void saveNetworkToJson(std::string &filename);
    void loadNetworkFromJson( std::string &filename);
};

#endif

/*
    // Topology -> Array of values that corrosponds to the size of neruons in each layer
    // [I]->[H]->[O]  topolgy =3, 1 input, 1 hidden, 1
    // The corrospoding weight matrices will be topology.size()-1 = 2
    // Each index of the topology = Corrospoding index of Layer's Index
*/