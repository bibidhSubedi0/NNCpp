#!/bin/bash
echo "Compiling..."
g++ -fdiagnostics-color=always -g network.cpp AllFiles/Layer.cpp AllFiles/Neuron.cpp AllFiles/Matrix.cpp AllFiles/NN.cpp -o run.exe
echo "Compilation Complete."
