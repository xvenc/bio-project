# Simple makefile to build shared library of our RLT algorithm to run from python
# Also contains main target to build main file used for debugging purposes

all: lib main

lib: repeatedlinetracking.cpp
		g++ -std=c++17 -O0 -g -fPIC -shared repeatedlinetracking.cpp `pkg-config --cflags --libs opencv4` -o librlt.so

main: main.cpp
		g++ -std=c++17 -O0 -g main.cpp `pkg-config --cflags --libs opencv4`  -o main -L. -lrlt
