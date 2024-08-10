#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include<iostream>
#include"Neuron.hpp"
#include<vector>
#include"Matrix.hpp"
using namespace std;

class Layer
{
    public:

    Layer(int size);
    void setVal(int index, double val);

    Matrix *convertTOMatrixVal();
    Matrix *convertTOMatrixActivatedVal();
    Matrix *convertTOMatrixDerivedVal();
    Layer *feedForward(Matrix *LastWeights,Matrix *LastBias,bool isFirst);

    vector<Neuron *> getNeurons();
    // Use weightMatrix to calculate Value at that Layer


    private:
    int size;
    vector<Neuron *> neurons;
};


#endif