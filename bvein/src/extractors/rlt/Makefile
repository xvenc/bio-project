# Simple makefile to build shared library of our RLT algorithm to run from python
lib: repeatedlinetracking.cpp
		g++ -std=c++17 -O3 -g -fPIC -shared repeatedlinetracking.cpp `pkg-config --cflags --libs opencv4` -o librlt.so
