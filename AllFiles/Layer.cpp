#include<math.h>
#include "NeuralNetwork.hpp"
using namespace std;

Layer :: Layer(int size)
{
    this->size =size;
    for(int i=0;i<size;i++)
    {
        Neuron *n = new Neuron(0.00);
        this->neurons.push_back(n);
    }
}

int Layer::getSize()
{
    return size;
}

Matrix *Layer::convertTOMatrixVal()
{
    Matrix *m = new Matrix(1,this->neurons.size(),false);
    for(int i=0;i<neurons.size();i++)
    {
        m->setVal(0,i,this->neurons[i]->getVal());
    }
    return m;
}
Matrix *Layer::convertTOMatrixActivatedVal()
{
    Matrix *m = new Matrix(1,this->neurons.size(),false);
        for(int i=0;i<neurons.size();i++)
    {
        m->setVal(0,i,this->neurons[i]->getActivatedVal());
    }
    return m;
}
Matrix *Layer::convertTOMatrixDerivedVal()
{
    Matrix *m = new Matrix(1,this->neurons.size(),false);
        for(int i=0;i<neurons.size();i++)
    {
        m->setVal(0,i,this->neurons[i]->getDerivedVal());
    }
    return m;
}

Layer *Layer::feedForward(Matrix* Weights, Matrix *bias, bool isFirst)
{
    
    Matrix *this_layer_val;
    if(isFirst)
    {
        this_layer_val = convertTOMatrixVal();
    }
    else{
        this_layer_val = convertTOMatrixActivatedVal();
    }
    
    Matrix *TransFormedMat = new Matrix(1,Weights->getNumCols(),false);
    Matrix *z = this_layer_val->Multiply(Weights);
    Matrix *zWithBias = z->Add(bias);
    
    // Put the Calculated Values in the Layer in form of vector rather then matrix
    Layer *temp = new Layer(Weights->getNumCols());
    for(int i=0;i<Weights->getNumCols();i++)
    {
        temp->setVal(i,zWithBias->getVal(0,i));
    }
    
    return temp;

}

vector<Neuron *> Layer::getNeurons()
{
    return neurons;
}

void Layer::setVal(int i, double v)
{
    this->neurons[i]->setVal(v);
}