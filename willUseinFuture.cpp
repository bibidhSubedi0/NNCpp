#include "AllFiles/NeuralNetwork.hpp"
#include <iostream>
#include <fstream>
using namespace std;

struct outputInformation
{
    double BestLearningRate;
    vector<int> bestTopology;
    vector<double> errorForAllCombinations;
    double leastError;
    NN *best_Network = NULL;
};

outputInformation TrainNetwork(vector<double> lrs, vector<vector<int>> topologies, vector<vector<double>> inputs, vector<vector<double>> targets, int totalEpoch)
{
    outputInformation result;

    double bestLeastError = 100;
    for (auto lr : lrs)
    {
        for (auto topo : topologies)
        {
            cout << "Current Learning Rate : " << lr << endl;
            ;
            cout << "Current Topology : ";
            for (int i = 0; i < topo.size(); i++)
            {
                cout << topo[i] << " ";
            }
            cout << endl;

            double errorForThisLrAndThisTopology = 0;

            NN *Network = new NN(topo, lr);
            int epoach = 0;

            while (epoach < totalEpoch)
            {
                epoach++;
                errorForThisLrAndThisTopology = 0.0;

                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    vector<double> input = inputs[i];
                    vector<double> target = targets[i];

                    Network->setCurrentInput(input);
                    Network->setTarget(target);

                    Network->forwardPropogation();

                    Network->setErrors();
                    Network->backPropogation();

                    errorForThisLrAndThisTopology += Network->getGlobalError();
                }

                errorForThisLrAndThisTopology = (errorForThisLrAndThisTopology / inputs[0].size());
            }

            if (errorForThisLrAndThisTopology < bestLeastError)
            {
                result.BestLearningRate = lr;
                result.leastError = errorForThisLrAndThisTopology;
                result.bestTopology = topo;
                bestLeastError = errorForThisLrAndThisTopology;
                result.best_Network = new NN(*Network);
            }
            result.errorForAllCombinations.push_back(errorForThisLrAndThisTopology);
            delete Network;
            cout << "Error for this configuration : " << errorForThisLrAndThisTopology << endl;
            cout << "---------------------------------------------------------------------------------\n";
        }
    }
    return result;
}

int main()
{
    vector<int> topology = {3, 4, 8, 16, 8, 8, 4, 3};

    double learningRate = 1;

    // Traing will be done for a given set of learning rates i.e. (1) vector<double> lrs, a set out input and targets obviously, a set of topologies
    vector<double> lrs = {0.25, 0.5, 1, 1.25};
    vector<vector<int>> topologies = {{3, 2, 3}, {3, 4, 4, 3}, {3, 4, 8, 4, 3}, {3, 4, 8, 16, 8, 4, 3}};
    vector<vector<double>> inputs = {{0, 0, 0},
                                     {0, 0, 1},
                                     {0, 1, 0},
                                     {0, 1, 1},
                                     {1, 0, 0},
                                     {1, 0, 1},
                                     {1, 1, 0},
                                     {1, 1, 1}};

    vector<vector<double>> targets = {{0, 0, 0},
                                      {0, 0, 1},
                                      {0, 1, 0},
                                      {0, 1, 1},
                                      {1, 0, 0},
                                      {1, 0, 1},
                                      {1, 1, 0},
                                      {1, 1, 1}};
    int totalEpoch = 2000;
    // Training Part
    outputInformation trained = TrainNetwork(lrs, topologies, inputs, targets, totalEpoch);

    // Results
    cout << "---------------------------------------------------------------------------------\n";
    cout << "-----------------------------Results---------------------------------------------\n";
    cout << "---------------------------------------------------------------------------------\n";
    cout << "Best Topology : ";
    for (int i = 0; i < trained.bestTopology.size(); i++)
    {
        cout << trained.bestTopology[i] << " ";
    }
    cout << endl;

    cout << "Best Learning Rate : " << trained.BestLearningRate << endl;
    cout << "Least Error : " << trained.leastError << endl;

    cout << "---------------------------------------------------------------------------------\n";
    cout << "Validation : " << endl;
    // Validation
    cout << "Input : " << endl;
    cout << "1\t0\t1\n";
    vector<double> input = {1, 0, 1};
    trained.best_Network->setCurrentInput(input);
    trained.best_Network->setTarget(input);
    trained.best_Network->forwardPropogation();
    trained.best_Network->setErrors();
    Matrix *output = trained.best_Network->GetLayer(trained.bestTopology.size() - 1)->convertTOMatrixActivatedVal();
    cout << "Output" << endl;
    output->printToConsole();
    cout << "Error" << endl;
    cout << trained.best_Network->getGlobalError() << endl;
    cout << endl;

    // cout << "All errors : " << endl;
    // for (int i = 0; i < trained.errorForAllCombinations.size(); i++)
    // {
    //     cout << trained.errorForAllCombinations[i] << endl;
    // }
}
