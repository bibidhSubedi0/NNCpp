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

Matrix *Layer::convertTOMatrixVal() //0 initilaized "neurons" in layer class is made a row matrix
{
    Matrix *m = new Matrix(1,this->neurons.size(),false);
    for(int i=0;i<neurons.size();i++)
    {
        m->setVal(0,i,this->neurons[i]->getVal());
    }
    return m;
}
Matrix *Layer::convertTOMatrixActivatedVal() //"neurons" from layer jun banne bittikai activate hunthe are connverted to row marix
{
    Matrix *m = new Matrix(1,this->neurons.size(),false);
    for(int i=0;i<neurons.size();i++)
    {
        m->setVal(0,i,this->neurons[i]->getActivatedVal());
    }
    return m;
}
Matrix *Layer::convertTOMatrixDerivedVal() //layer bata access garincha yo function, derived value of neurons lai matrix banauna
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




    Matrix *z = *this_layer_val*Weights;
    Matrix *zWithBias = *z+bias;
    
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