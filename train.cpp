#include "AllFiles/NeuralNetwork.hpp"

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
            cout << "Current Topology : ";
            for (int i = 0; i < topo.size(); i++)
            {
                cout << topo[i] << " ";
            }
            cout << endl;

            long double errorForThisLrAndThisTopology = 0.00;

            

            NN *Network = new NN(topo, lr, inputs.size());

            
            int epoch = 0;


            // size_t totalElements=0;
            // for(auto & input: inputs)
            // {
            //     totalElements+=input.size();
            // }


            while (epoch < totalEpoch)
            {
                epoch++;
               errorForThisLrAndThisTopology = 0.0;

                for (size_t i = 0; i < inputs.size(); ++i)
                {
                   
                    vector<double> input = inputs[i]; 
                    vector<double> target = targets[i];

                    Network->setCurrentInput(input);
                    Network->setTarget(target);

                    Network->forwardPropogation();
                    // Network->printToConsole();

                    // Network->printWeightMatrices();
                    // Network->printBiases();
                    //cout<<Network->lastEpochError()<<endl; //Network banne bittikai histError 1 push garda balla yesle 1 dincha natra segmentation fault

                    //cout<<"hi"<<endl;
                    Network->setErrors();
                    //Network->printErrors();

                    cout<<"above"<<endl; 
                    Network->backPropogation();
                    cout<<"below"<<endl;
                    //Network->printGradientsAccumulator();

                    errorForThisLrAndThisTopology += Network->getGlobalError();
                    cout<<"aye ma tw"<<endl;

                    // errorForThisLrAndThisTopology = errorForThisLrAndThisTopology * (inputs.size() + 1) + Network->getGlobalError();

                    cout<<errorForThisLrAndThisTopology<<endl;
                }
                
                cout<<"abcd"<<endl;
                Network->printGradientsAccumulator();
                cout<<"abcddddddddddddddddd"<<endl;
                Network->gradientDescent();

                errorForThisLrAndThisTopology = errorForThisLrAndThisTopology / (inputs.size() + 1 ) ;
                //errorForThisLrAndThisTopology = errorForThisLrAndThisTopology / (inputs.size() + 1 ) ;

                             
            }

            // errorForThisLrAndThisTopology = (errorForThisLrAndThisTopology / (inputs.size() )) ;

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

