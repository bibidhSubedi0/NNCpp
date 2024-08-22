#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include<iostream>
#include"Neuron.hpp"
#include<vector>
using namespace std;

class Matrix
{
    public:

    Matrix(int numRows, int numCols, bool isRandom);
    double genRandomNumber();
    Matrix *tranpose();
    Matrix *Multiply(Matrix *& A);
    Matrix *operator +(Matrix *& A);


    void setVal(int r, int c, double v);
    double getVal(int r, int c);

    void printToConsole();

    int getNumRow(){return this->numRows;}
    int getNumCols(){return this->numCols;}

    private:
    int numRows;
    int numCols;
    vector<vector<double>> values;
};


#endif