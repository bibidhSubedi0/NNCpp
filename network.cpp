#include "AllFiles/NeuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include "train.cpp"
using namespace std;


int main()
{



    // Prameters for the neural network to be trained on.......
    vector<double> lrs = {1,0.5,4,10,100};
    vector<vector<int>> topologies = {{3, 4, 8, 4, 3},{3,4,3},{3,2,3}};
    vector<vector<double>> inputs = {{1,0,1}};//,{1,1,1}};,{0,1,1}};
    vector<vector<double>> targets = {{1,0,1}};//,{1,1,1}};,{0,1,1}};
    int totalEpoch = 1200;





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


     cout<<"Final output of the Network : "<<endl;

    // trained.best_Network->printToConsole();

    cout<<endl;

    trained.best_Network->printHistErrors();

    trained.best_Network->saveHistErrors();

    cout << "\n---------------------------------------------------------------------------------\n";
 
    
    
}
