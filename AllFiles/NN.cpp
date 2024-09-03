#include "NeuralNetwork.hpp"
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <fstream>

long double NN::getGlobalError()
{
    return this->error;
}

long double NN::lastEpoachError()
{
    return histErrors[histErrors.size() - 1];
}

void NN::printHistErrors()
{
    cout<<"\n Printing Errors from all epochs:"<<endl;
    for (int i = 0; i < this->histErrors.size(); i++)
    {
        cout <<"Epoch: "<<i<<", Error:"<< histErrors.at(i) << " \n ";
    }
}

void NN:: saveHistErrors()
{
    ofstream outFile("error_vs_epoch.csv");

    if (outFile.is_open()) {

        outFile << "Epoch,Error\n";

        for (size_t i = 0; i < this->histErrors.size(); ++i) {
            outFile << i << "," << histErrors[i] << "\n";
        }
    outFile.close();
    } 
    else {
        cerr << "Unable to open file for writing.\n";
    }

}

vector<long double> NN::gethisterrors()
{
    return histErrors;
}

long double NN::getLearningRate()
{
    return learningRate;
}

NN ::NN(vector<int> topology, double lr)
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
        this->BaisMatrices.push_back(mb);
    }
    histErrors.push_back(1);
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
    for (int i = 0; i < input.size(); i++)
    {
        cout << input.at(i) << "\t\t";
    }
    cout << endl;

    // Print the outputs to the network
    for (int i = 0; i < layers.at(layers.size() - 1)->getSize(); i++)
    {
        cout << layers.at(layers.size() - 1)->getNeurons().at(i)->getActivatedVal() << "\t";
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
        layers[i + 1] = layers[i]->feedForward(weightMatrices[i], BaisMatrices[i], (i == 0));
    }
}

void NN::printBiases()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Bias for Hidden Layer : " << i + 1 << endl;
        BaisMatrices[i]->printToConsole();
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
    errorDerivatives.resize(errors.size());
    for (int i = 0; i < target.size(); i++)
    {
        double req = target[i];
        double act = outputNeurons[i]->getActivatedVal();
        this->errors[i] = 0.5 * pow(abs((req - act)), 2);
        errorDerivatives[i] = act - req;
        this->error += errors[i];
    }

    this->histErrors.push_back(this->error);
}

void NN::backPropogation()

{
    // Okay lets fucking organize this little pice of shit code

    // All the Matrices we will need thorughout the process :
    vector<Matrix *> newWeights;
    Matrix *deltaWeights;
    Matrix *gradients;
    Matrix *DerivedValuesFromOtoH;
    Matrix *gradientsTransposed;
    Matrix *PreviousLayerActivatedVals;
    Matrix *tempNewWeights;
    Matrix *lastGradient;
    Matrix *tranposedWeightMatrices;
    Matrix *hiddenDerived;
    Matrix *transposedHidden;

    // Index of the outermost layer, as the name suggests....
    int outputLayerIndex = this->topology.size() - 1;

    // A graident as used in tyo vector calculus was chapter, gives the rate of change of a function Ψ  in a normal direction
    // But here we used to get the rate of change of the error and its direction, as in +ve error or -ve error, the formula for this gradient
    // used here is G [(e1' x y1')  (e2' x y2') ........ (en' x yn')] where en' is the value of error derivate for the nth neuron and yn' is the derived value for that neuron
    // Formula ka bato ayo tha xaina, dont ask, just use it
    gradients = new Matrix(
        1,
        this->topology.at(outputLayerIndex),
        false);

    // Well As the name suggests, just the derived values at output layer needed for gradient
    DerivedValuesFromOtoH = this->layers.at(outputLayerIndex)->convertTOMatrixDerivedVal();

    for (int i = 0; i < this->topology.at(outputLayerIndex); i++)
    {
        double e = this->errorDerivatives.at(i);
        double y = DerivedValuesFromOtoH->getVal(0, i);
        double g = e * y;
        gradients->setVal(0, i, g);
    }

    // Transpoing the gradient cuz the plot demands
    gradientsTransposed = gradients->tranpose();

    // ------------------------ So upto here the gradient has been calculated --------------------------
    // -------------------------for output to first hidden only btw ------------------------------------

    PreviousLayerActivatedVals = this->layers.at(outputLayerIndex - 1)->convertTOMatrixActivatedVal();

    // Now that we have the gradient, i.e.direction of the error function of the network, as in we know, how the error function is changing, as in increasing or decreasing at that point
    // We can use it calculate new weights, but first we need to calculate the chagne in the weights, i.e. ..... you know it..... comon say it...... YESS......DeltaWeight which is given by
    // δW = Transpose((Transpose(G) * Z)), where
    // G is obviouslt the gradient and Z is the previous/Left layer's activated values, becuase as we know, these activated values from prevoius layer, determine the new values of the current layer
    // Think of it as a chain effect, tyo partial derivates ma chain rule lagaya jastai

    deltaWeights = new Matrix(
        gradientsTransposed->getNumRow(),
        PreviousLayerActivatedVals->getNumCols(),
        false);

    deltaWeights = *gradientsTransposed*PreviousLayerActivatedVals;

    // Now new weights is simply given by Previous weight - DeltaWeright for each value of weight between those 2 layers
    // We can add the learning rate here as learning rate simply means the rate at which the weights will be changed

    tempNewWeights = new Matrix(
        this->topology.at(outputLayerIndex - 1),
        this->topology.at(outputLayerIndex),
        false);

    for (int r = 0; r < this->topology.at(outputLayerIndex - 1); r++)
    {
        for (int c = 0; c < this->topology.at(outputLayerIndex); c++)
        {

            double originalValue = this->weightMatrices.at(outputLayerIndex - 1)->getVal(r, c);
            double deltaValue = deltaWeights->getVal(c, r);
            deltaValue = this->learningRate * deltaValue;

            tempNewWeights->setVal(r, c, (originalValue - deltaValue));
        }
    }

    newWeights.push_back(new Matrix(*tempNewWeights));

    // Hya samma tai new weight calculate gareo but just for output layer to first hidden layer

    // Little bit of memory management cuz, you know, ima big boiiii now....
    delete gradientsTransposed;
    delete PreviousLayerActivatedVals;
    delete tempNewWeights;
    delete deltaWeights;
    delete DerivedValuesFromOtoH;

    // Now aila samma ta just Output to last hidden layer gareko, now need to do from last to first hidden layer

    for (int i = (outputLayerIndex - 1); i > 0; i--)
    {

        // Here Pretty much everything is same except, we use a seperate formula to calculat the gradient kina vane last layer ko gradient depended on nothing but the output
        // But second last, third last gardai, upto fist layer ko gardient WILL DEPEND UPON the gradient that comes after them,
        // Jati thulo guff gareni at the end of the day tai chain rule nai applay garya ho to calculate the graident in this layer using last calculated gardient
        // Tyo vanda badi maile ni CS/Math ko PHD gareko xaina, dont fucking ask
        // Anyways the formula is Gradient(G) = (LastGradient x Transpose(WeightForThatLayer)) x LastLayerDerivativeMatrices

        // Now i know the word 'last' maybe confusing here, as i using it to descibe both 'previous' and 'next' in different context but
        // Pulchowk ma computer engineering padxu, you can figure that out

        // Just tai copy construct use gareko, we cant just assign newGradient to graident, cuz tyo sab pointers ho and you will just be keeping the
        // pointer to the graident in newGraidents and when you delete graidents, newGradient will become dangling pointer
        // Tai vara need to allocate seperate memory for newGradient, just a bit of technical knowladge (I SPENT ETERNITY TRYING TO FIGURE THIS BITCH OUT T^T)
        lastGradient = new Matrix(*gradients);
        delete gradients;

        // If you have not already figured out, we tranpose matrices whenever the fuck we want/need
        tranposedWeightMatrices = this->weightMatrices.at(i)->tranpose();

        // Again Same thing with gradient as before
        gradients = new Matrix(
            lastGradient->getNumRow(),
            tranposedWeightMatrices->getNumCols(),
            false);

        gradients = *lastGradient*tranposedWeightMatrices;

        hiddenDerived = this->layers.at(i)->convertTOMatrixDerivedVal();

        for (int colCounter = 0; colCounter < hiddenDerived->getNumCols(); colCounter++)
        {
            double g = gradients->getVal(0, colCounter) * hiddenDerived->getVal(0, colCounter);
            gradients->setVal(0, colCounter, g);
        }

        // Hya samma tai ni gareko ho, just what i descibed above, tellai code gareko
        // Now for delta weight, not exactly same as before but similer
        // δW = Transpose(PreviousLayerActivaedVals) * Gradients which we jsut calculated

        if (i == 1)
        {
            PreviousLayerActivatedVals = this->layers.at(0)->convertTOMatrixVal();
        }
        else
        {
            PreviousLayerActivatedVals = this->layers.at(i - 1)->convertTOMatrixActivatedVal();
        }

        transposedHidden = PreviousLayerActivatedVals->tranpose();

        deltaWeights = new Matrix(
            transposedHidden->getNumRow(),
            gradients->getNumCols(),
            false);

        deltaWeights = *transposedHidden*gradients;

        tempNewWeights = new Matrix(
            this->weightMatrices.at(i - 1)->getNumRow(),
            this->weightMatrices.at(i - 1)->getNumCols(),
            false);

        // And updating new weights is the exact same shit,
        // New weight =  old weight - change in weight
        // also learning rate is incorporated as before

        for (int r = 0; r < tempNewWeights->getNumRow(); r++)
        {
            for (int c = 0; c < tempNewWeights->getNumCols(); c++)
            {
                double originalValue = this->weightMatrices.at(i - 1)->getVal(r, c);
                double deltaValue = deltaWeights->getVal(r, c);

                deltaValue = this->learningRate * deltaValue;

                tempNewWeights->setVal(r, c, (originalValue - deltaValue));
            }
        }

        newWeights.push_back(new Matrix(*tempNewWeights));

        // Bit more of memroy management

        delete lastGradient;
        delete tranposedWeightMatrices;
        delete hiddenDerived;
        delete PreviousLayerActivatedVals;
        delete transposedHidden;
        delete tempNewWeights;
        delete deltaWeights;
    }
    delete gradients;

    for (int i = 0; i < this->weightMatrices.size(); i++)
    {
        delete this->weightMatrices[i];
    }

    this->weightMatrices.clear();

    // We reverse becuase, well think about it, we are moving from out to in,
    // so we were adding the weights for the outermost layers first, so
    // we reverse to make outmost weight matrices, well......, goto outermost layer,... duh
    reverse(newWeights.begin(), newWeights.end());

    // assigning newWeights to weightMatrices and the back prop is over
    weightMatrices = newWeights;
}

