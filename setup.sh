#!/bin/bash

# First, install the required packages
pip install -r requirements.txt

# Build the RLT shared library
cd bvein/src/extractors/rlt
make