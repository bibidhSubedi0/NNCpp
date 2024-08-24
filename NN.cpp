#include "NN.hpp"
#include <assert.h>
#include <cmath>
#include <algorithm>

#include "json.hpp"
#include <fstream>



double NN::getGlobalError()
{
    return this->error;
}

double NN::lastEpoachError()
{
    return histErrors[histErrors.size()-1];
}

void NN::printHistErrors()
{
    for(int i=0;i<this->histErrors.size();i++)
    {
        cout<<histErrors.at(i)<<" , ";
    }
}

double NN::getLearningRate()
{
    return learningRate;
}


NN ::NN(vector<int> topology,double lr)
{
    this->learningRate = lr;
    this->topology = topology;
    this->topologySize = topology.size();
    for (int i = 0; i < topologySize; i++)
    {
        Layer *l = new Layer(topology[i]);
        this->layers.push_back(l);
    }

    for (int i = 0; i < topologySize - 1; i++)
    {
        Matrix *mw = new Matrix(topology[i], topology[i + 1], true);
        this->weightMatrices.push_back(mw);

        Matrix *mb = new Matrix(1, topology[i + 1], false);
        this->biasMatrices.push_back(mb);
    }
}

void NN::setTarget(vector<double> target)
{
    this->target = target;
}

void NN::printErrors()
{
    // cout<<"This Iteration Error"<<endl;
    // for(auto err : this->errors)
    // {
    //     cout<<"Ex : "<<err<<"  ";
    // }
    // cout<<endl;

    // cout<<"Historical Errors"<<endl;
    // for(auto err : this->errors)
    // {
    //     cout<<"Eh : "<<err<<"  ";
    // }
    // cout<<endl;

    cout << "Total Error : " << this->error << endl;
}

void NN::setCurrentInput(vector<double> input)
{
    this->input = input;

    for (int i = 0; i < input.size(); i++)
    {
        this->layers[0]->setVal(i, input[i]);
    }
}
void NN::printToConsole()
{

    // Print the inputs to the network
    for(int i=0;i<input.size();i++)
    {
        cout<<input.at(i)<<"\t\t";
    }
    cout<<endl;

    // Print the outputs to the network
    for(int i=0;i<layers.at( layers.size()-1)->getSize();i++)
    {
        cout<<layers.at(layers.size()-1)->getNeurons().at(i)->getActivatedVal()<<"\t";
    }

}

Layer *NN::GetLayer(int nth)
{
    return layers[nth];
}

void NN::printWeightMatrices()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Weights for Hidden Layer : " << i + 1 << endl;
        weightMatrices[i]->printToConsole();
    }
}

void NN::forwardPropogation()
{
    for (int i = 0; i < layers.size() - 1; i++)
    {
        layers[i + 1] = layers[i]->feedForward(weightMatrices[i], biasMatrices[i], (i == 0));
    }
}

void NN::printBiases()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Bias for Hidden Layer : " << i + 1 << endl;
        biasMatrices[i]->printToConsole();
    }
}

void NN::setErrors()
{
    if (this->target.size() == 0)
    {
        cerr << "No target found" << endl;
        assert(false);
    }

    if (target.size() != layers[layers.size() - 1]->getNeurons().size())
    {
        cerr << "The size of the target is not equal to the size of the output" << endl;
        assert(false);
    }

    errors.resize(target.size());

    this->error = 0;
    int outputLayerIndx = this->layers.size() - 1;
    vector<Neuron *> outputNeurons = this->layers[outputLayerIndx]->getNeurons();
    for (int i = 0; i < target.size(); i++)
    {
        double terr = (outputNeurons[i]->getActivatedVal() - target[i]);
        errors[i] = terr;
        this->error += pow(terr,2);
    }
    this->error=0.5 * this->error;
    this->histErrors.push_back(this->error);
}



void NN::backPropogation()
{
    vector<Matrix *> newWeights;

    Matrix *gardient;
    // Output to first hidden from back
    int outputLayerIndex = this->layers.size() - 1;

    // Y->Z means output to hidden, basically gets the derived values at output layer,
    //  i.e. change in the output at last layer with respcet to weights at that layer
    Matrix *DerivedValuesFromYtoZ = this->layers[outputLayerIndex]->convertTOMatrixDerivedVal();

    // GradientYtoZ is just a matrix with the dl/dw
    Matrix *GraidentYtoZ = new Matrix(1, this->layers[outputLayerIndex]->getNeurons().size(), false);

    for (int i = 0; i < this->layers.size(); i++)
    {
        double d = DerivedValuesFromYtoZ->getVal(0, i);
        double e = this->errors[i];
        double g = d * e;
        GraidentYtoZ->setVal(0, i, g);
    }

    int lastHiddenLayerIdx = outputLayerIndex - 1;
    Layer *lastHiddenLayer = this->layers[lastHiddenLayerIdx];

    Matrix *weightsOutputToHidden = this->weightMatrices[outputLayerIndex - 1];
    // Matrix *deltaOutputHidden = new Matrix(weightsOutputToHidden->getNumRows(),weightsOutputToHidden->getNumCols(),false);
    Matrix *deltaOutputHidden = GraidentYtoZ->transpose();

    Matrix *LastHiddenLayerActivatedVals = lastHiddenLayer->convertTOMatrixActivatedVal();
    deltaOutputHidden = deltaOutputHidden->Multiply(LastHiddenLayerActivatedVals);
    deltaOutputHidden = deltaOutputHidden->transpose();

    Matrix *newWeightsOutputToHidden = new Matrix(deltaOutputHidden->getNumRows(), deltaOutputHidden->getNumCols(), false);

    for (int r = 0; r < deltaOutputHidden->getNumRows(); r++)
    {
        for (int c = 0; c < deltaOutputHidden->getNumCols(); c++)
        {
            double orgWeight = weightsOutputToHidden->getVal(r, c);
            double delWeight = deltaOutputHidden->getVal(r, c);
            newWeightsOutputToHidden->setVal(r, c, (orgWeight - delWeight));
            // For learning rate
            newWeightsOutputToHidden->setVal(r,c,(newWeightsOutputToHidden->getVal(r,c))*learningRate);
        }
    }

    newWeights.push_back(newWeightsOutputToHidden);

    gardient = new Matrix(GraidentYtoZ->getNumRows(), GraidentYtoZ->getNumCols(), false);

    for (int r = 0; r < GraidentYtoZ->getNumRows(); r++)
    {
        for (int c = 0; c < GraidentYtoZ->getNumCols(); c++)
        {
            gardient->setVal(r, c, GraidentYtoZ->getVal(r, c));
        }
    }

    //                  Safe ---
    for (int i = outputLayerIndex - 1; i > 0; i--)
    {
        // Moving from last hidden layer down to input layer
        // Delta Weights for the hidden output layer

        Layer *l = this->layers[i];

        // Derived values at Layer L i.e. f'(wij)
        Matrix *derivedHidden = l->convertTOMatrixDerivedVal();
        Matrix *deriveGraident = new Matrix(1, l->getNeurons().size(), false);
        Matrix *activatedHidden = l->convertTOMatrixActivatedVal();
        Matrix *weightMatrix = this->weightMatrices[i];
        Matrix *origianlWeight = this->weightMatrices[i - 1];

        for (int r = 0; r < weightMatrix->getNumRows(); r++)
        {
            double sum = 0;
            for (int c = 0; c < weightMatrix->getNumCols(); c++)
            {
                double p = gardient->getVal(0, c) * weightMatrix->getVal(r, c);
                sum += p;
            }
            double g = sum * activatedHidden->getVal(0, r);
            deriveGraident->setVal(0, r, g);
        }

        // deriveGraident = deriveGraident->transpose();

        Matrix *leftNeuronsMatrix = (i - 1) == 0 ? this->layers[0]->convertTOMatrixVal() : this->layers[i - 1]->convertTOMatrixActivatedVal();
        // Matrix *leftNeuronsMatrix = this->layers[i]->convertTOMatrixActivatedVal();

        Matrix *deltaWeights = deriveGraident->transpose();
        deltaWeights = deltaWeights->Multiply(leftNeuronsMatrix);
        deltaWeights = deltaWeights->transpose();

        Matrix *newWeightsHidden = new Matrix(deltaWeights->getNumRows(), deltaWeights->getNumCols(), false);

        for (int r = 0; r < newWeightsHidden->getNumRows(); r++)
        {
            for (int c = 0; c < newWeightsHidden->getNumCols(); c++)
            {
                double w = origianlWeight->getVal(r, c);
                double d = deltaWeights->getVal(r, c);
                double n = w - d;
                newWeightsHidden->setVal(r, c, n);
            }
        }

        //                      -- Safe

        gardient = new Matrix(deriveGraident->getNumRows(), deriveGraident->getNumCols(), false);
        for (int r = 0; r < deriveGraident->getNumRows(); r++)
        {
            for (int c = 0; c < deriveGraident->getNumCols(); c++)
            {
                gardient->setVal(r, c, deriveGraident->getVal(r, c));
            }
        }

        newWeights.push_back(newWeightsHidden);
    }

    // Supposed to be a } here?

    std::reverse(newWeights.begin(), newWeights.end());

    this->weightMatrices = newWeights;
}

using json = nlohmann::json;

void NN::saveNetworkToJson(std::string &filename) {
    // initializing JSON object so as to use it save the network 
    json networkJson;

    // saves the topology size in JSON file under the key "topology"
    networkJson["topology"] = topology;

    // save the weight matrices
    for (int i = 0; i < weightMatrices.size(); i++) {
        // done to make a array for each weight matrix
        json weightMatrixJson = json::array();
        for (int j = 0; j < weightMatrices[i]->getNumRows(); j++) {
            json rowJson = json::array();
            for (int k = 0; k < weightMatrices[i]->getNumCols(); k++) {
                rowJson.push_back(weightMatrices[i]->getVal(j, k));
            }
            weightMatrixJson.push_back(rowJson);
        }
        networkJson["weights"][i] = weightMatrixJson;
    }

    // save the bias matrices
    for (int i = 0; i < biasMatrices.size(); i++) {
        json biasMatrixJson = json::array();
        for (int r = 0; r < biasMatrices[i]->getNumRows(); r++) {
            json rowJson = json::array();
            for (int c = 0; c < biasMatrices[i]->getNumCols(); c++) {
                rowJson.push_back(biasMatrices[i]->getVal(r, c));
            }
            biasMatrixJson.push_back(rowJson);
        }
        // saves the biasMatrices in JSON file under the key "biases"
        networkJson["biases"][i] = biasMatrixJson;
    }

    std::ofstream outFile(filename);
    // dump() is used to convert the JSON object to string 
    // saving the JSON object to the file
    outFile << networkJson.dump(4);
    outFile.close();

    std::cout << "Network saved to " << filename << " as JSON." << std::endl;
}

void NN::loadNetworkFromJson(std::string &filename) {
    // creating a ifstream object to read the file
    std::ifstream inFile(filename);
    // check if the file is open or not using the is_open() function provided by the ifstream class
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file for loading network!" << std::endl;
        return;
    }

    json networkJson;
    inFile >> networkJson;
    inFile.close();

    // clear the existing topology size
    topology.clear();
    // retrieveing the topology size from the JSON file
    topology = networkJson["topology"].get<std::vector<int>>();

    // Rebuild network structure with new topology
    topologySize = topology.size();
    // layers is vector of Layer pointers
    layers.clear();
    weightMatrices.clear();
    biasMatrices.clear();

    for (int i = 0; i < topologySize; i++) {
        Layer *l = new Layer(topology[i]);
        layers.push_back(l);
    }

    for (int i = 0; i < topologySize - 1; i++) {
        Matrix *mw = new Matrix(topology[i], topology[i + 1], false);
        weightMatrices.push_back(mw);

        Matrix *mb = new Matrix(1, topology[i + 1], false);
        biasMatrices.push_back(mb);
    }

    // Load weight matrices
    for (int i = 0; i < weightMatrices.size(); i++) {
        // here the values under the key "weights" of i index are stored in weightMatrixJson array
        json weightMatrixJson = networkJson["weights"][i];
        for (int j= 0; j< weightMatrices[i]->getNumRows(); j++) {
            for (int k = 0; k< weightMatrices[i]->getNumCols(); k++) {
                weightMatrices[i]->setVal(j, k, weightMatrixJson[j][k]);
            }
        }
    }

    
    // Load bias matrices
    for (int i = 0; i < biasMatrices.size(); i++) {
        json biasMatrixJson = networkJson["biases"][i];
        for (int j= 0; j< biasMatrices[i]->getNumRows(); j++) {
            for (int k = 0; k< biasMatrices[i]->getNumCols(); k++) {
                biasMatrices[i]->setVal(j, k, biasMatrixJson[j][k]);
            }
        }
    }

    // printing the weight matrices
    cout << "Weight Matrices" << endl;
    for (int i = 0; i < weightMatrices.size(); i++) {
        cout << "Weight Matrix " << i << endl;
        weightMatrices[i]->printToConsole();
        cout << endl;
    }

    // printing the weight matrices
    cout << "Bias Matrices" << endl;
    for (int i = 0; i < biasMatrices.size(); i++) {
        cout << "Bias Matrix " << i << endl;
        biasMatrices[i]->printToConsole();
        cout << endl;
    }
}
