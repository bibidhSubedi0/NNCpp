#!/bin/bash
echo "Compiling..."
g++ -fdiagnostics-color=always -g main.cpp Layer.cpp Neuron.cpp Matrix.cpp NN.cpp -o run
echo "Compilation complete."