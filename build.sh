#!/bin/bash

# Find all .cpp files in the current directory and subdirectories
SRCS=$(find . -name "*.cpp")

# Compile and link all .cpp files into an executable named 'main'
# g++ -lraylib \
#     -lGL \
#     -lm \
#     -lpthread \
#     -ldl \
#     -lrt \
#     -lX11 \
#     -lstdc++ \

 g++   -fdiagnostics-color=always \
    -g \
    ${SRCS} \
    -o  output

