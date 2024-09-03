#ifndef _NN_HPP_
#define _NN_HPP_

#include <iostream>
#include "Matrix.hpp"
#include "Layer.hpp"
#include <vector>
using namespace std;

class NN
{
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
    long double getGlobalError();
    long double lastEpoachError();
    void printHistErrors();
    void saveHistErrors();
    long double getLearningRate();
    void setErrorDerivatives();
    vector<long double> gethisterrors();

   

private:
    int topologySize;
    vector<int> topology;
    vector<Layer *> layers;
    vector<Matrix *> weightMatrices;
    vector<Matrix *> GradientMatrices;
    vector<double> input;
    vector<Matrix *> BaisMatrices;
    double error;
    vector<double> target;
    vector<long double> errors;
    vector<long double> histErrors;
    vector<double> errorDerivatives;
    double learningRate;

    

};

#endif
