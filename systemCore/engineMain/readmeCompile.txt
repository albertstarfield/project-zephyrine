# how to compile the cmake
cmake -B build -S . && cmake --build build

# make venv
python3 -m venv myenv
source myenv/bin/activate

# compile the g++
g++ -o main main.cpp

# clean the shit up
cd /home/adri/project-zephyrine/systemCore/engineMain/build
make clean
cmake ..
make
