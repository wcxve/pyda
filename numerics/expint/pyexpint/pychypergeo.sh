#!/bin/sh

# switch to anaconda 3.5
export PATH=/home/gui/anaconda3/bin:$PATH

# project parameters
PROJECT="expint"
PROJECT_ROOT="../$PROJECT"
ANACONDA="/home/gui/anaconda3/include/python3.5m/"
SOURCE_DIR="$PROJECT_ROOT/src"
INCLUDE_DIR="$PROJECT_ROOT/include"

# swig
echo "generating swig wrapper..."
swig -c++ -python expint.i

#compile
echo "compiling chypergeo library..."
g++ -c -O3 -fopenmp -lquadmath -W -fpic -std=gnu++11 $SOURCE_DIR/*.cpp expint_wrap.cxx -I $ANACONDA -I $INCLUDE_DIR

#build library
echo "building python module..."
g++ -shared -fopenmp *.o -lquadmath -o  _expint.so
