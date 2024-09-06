#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include<iostream>
using namespace std;

class Neuron
{
    public:

    Neuron(double val);
    void setVal(double v);
    // Activation Function

    //Fast Sigmoid Function f(x) = x/(1+|x|)
    // Derivative => f'(x) = f(x) *(1-f(x))
    void Activate();
    void Derive();

    // Getter
    double getVal(){return this->val;}
    double getActivatedVal(){return this->activatedVal;}
    double getDerivedVal(){return this->derivedVal;}

    private:
    double val;
    double activatedVal; // After passing through sigmoid
    double derivedVal; // approx derivative of activacted val

};


#endif