#include "NeuralNetwork.hpp"
#include <random>

Matrix ::Matrix(int numRows, int numCols, bool isRandom = true)
{
    this->numRows = numRows;
    this->numCols = numCols;

    for (int i = 0; i < numRows; i++)
    {
        vector<double> rows;
        for (int j = 0; j < numCols; j++)
        {
            double r = 0.00;
            if (isRandom)
                r = this->genRandomNumber();
            rows.push_back(r);
        }
        this->values.push_back(rows);
    }
}


double Matrix::genRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister 19937 generator seeded with rd

    // Define the distribution for floating point numbers between 0 and 1
    std::uniform_real_distribution<float> dis(0.45f, 0.55f);

    // Generate a random float number between 0 and 1 with 3 decimal digits
    float random_number = dis(gen);
    return random_number;
    // Output the generated random number
    //return 0.5;
}

void Matrix ::printToConsole()
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            cout << this->values[i][j] << "\t";
        }
        cout << endl;
    }
}

void Matrix::setVal(int r, int c, double val)
{
    this->values[r][c] = val;
}

double Matrix::getVal(int r, int c)
{
    return this->values[r][c];
}

Matrix *Matrix ::tranpose()
{
    Matrix *tans = new Matrix(this->numCols, this->numRows, false);
    for (int orgRow = 0; orgRow < this->numRows; orgRow++)
    {
        for (int orgCol = 0; orgCol < this->numCols; orgCol++)
        {
            tans->values[orgCol][orgRow] = getVal(orgRow, orgCol);
        }
    }
    return tans;
}
Matrix *Matrix::operator *(Matrix *&B)
{
    int rows_A = numRows;
    int cols_A = numCols;
    int cols_B = B->getNumCols();
    int rows_B = B->getNumRows();
    // A vaneko Aafu, B vaneko arko
    // Resultant matrix C with size rows_A x
    Matrix *C = new Matrix(rows_A, cols_B, false);

    if (cols_A != rows_B)
    {
        std::cout << "-------------------------------------------------------------" << std::endl;
        std::cout << "Invlaid Dimensions" << std::endl;
        cout<<endl;
        cout<<endl;
        cout<<" This Matrix "<<endl;
        printToConsole();

        cout<<" Passed Matrix "<<endl;
        B->printToConsole();
        cout<<endl;
        cout<<endl;
        
    return C;
    }

    for (int i = 0; i < rows_A; ++i)
    {
        for (int j = 0; j < cols_B; ++j)
        {
            for (int k = 0; k < cols_A; ++k)
            {
                C->values[i][j] += values[i][k] * B->values[k][j];
            }
        }
    }

    return C;
}

Matrix *Matrix::operator +(Matrix *&B)
{
    
    int rows = B->getNumRows();
    int cols = B->getNumCols(); // Assuming both matrices have the same dimensions
    Matrix *ans = new Matrix(rows, cols, false);


    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            ans->values[i][j] = values[i][j] + B->values[i][j];
        }
    }

    return ans;
}

Matrix* Matrix::operator *(double scalar)
{
    Matrix* result= new Matrix(numRows, numCols, false); 

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            result->values[i][j] = values[i][j] * scalar; 
        }
    }

    return result;
}
