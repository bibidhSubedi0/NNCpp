#include"Neuron.hpp"
#include<math.h>
using namespace std;

Neuron :: Neuron(double val)
{
    this->val =val;
    Activate();
    Derive();
}

void Neuron::setVal(double val)
{
    this->val =val;
    Activate();
    Derive();
}

void Neuron :: Activate()
{
    //Fast Sigmoide Function f(x) = x/(1+|x|)
    this->activatedVal = this->val / (1+abs(this->val));   
}

void Neuron :: Derive()
{
    // Derivative => f'(x) = f(x) *(1-f(x))
    this->derivedVal = this->activatedVal * (1-this->activatedVal);
}