#include "AllFiles/NeuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include "train.cpp"
using std::cout;


int main()
{



    // Prameters for the neural network to be trained on.......
    vector<double> lrs = {1,0.1,0.5,0.2};
    vector<vector<int>> topologies = {{3,4,3},{3,2,3},{3,4,5,3}};
    vector<vector<double>> inputs = {{1,0,1},{1,1,1},{0,1,1}};//{1,1,0}};
    vector<vector<double>> targets = {{1,0,1},{1,1,1},{0,1,1}};//{1,1,0}};
    int totalEpoch = 1200;





    // Training Part
    outputInformation trained = TrainNetwork(lrs, topologies, inputs, targets, totalEpoch);





    // Results
    cout<< "---------------------------------------------------------------------------------\n";
    cout<< "-----------------------------Results---------------------------------------------\n";
    cout<< "---------------------------------------------------------------------------------\n";
    cout<< "Best Topology : ";
    for (int i = 0; i < trained.bestTopology.size(); i++)
    {
        cout<< trained.bestTopology[i] << " ";
    }
    cout<< endl;

    cout<< "Best Learning Rate : " << trained.BestLearningRate << endl;
    cout<< "Least Error : " << trained.leastError << endl;


     cout<<"Final output of the Network : "<<endl;

      

    //trained.best_Network->printHistErrors();

    trained.best_Network->saveHistErrors();

    cout<<endl;

    trained.best_Network->printToConsole();

    cout<< "\n---------------------------------------------------------------------------------\n";
 
    
    
}
