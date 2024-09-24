#!/bin/sh

rm -rf build
cmake -B build -S . && cmake --build build
./build/main