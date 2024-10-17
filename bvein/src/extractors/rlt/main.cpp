// This file is for debugging purposes only, it is not used in python

#include "repeatedlinetracking.cpp"

int main() {
    char* img = "image.png";
    char* mask = "mask.png";
    char* outfile = "test.png";

    int iter = 1000;
    int r = 1;
    int profile_w = 21;

    RepeatedLineTracking(img, mask, outfile, iter, r, profile_w);
}