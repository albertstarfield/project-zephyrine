#!/bin/bash

set -e

# Initialize Variables and check the platform and architecture
platform=$(uname -s)
arch=$(uname -m)
# Save the current working directory to "rootdir" variable (compensate spaces)
rootdir="$(pwd)"

if [ -z "${ENFORCE_NOACCEL}" ]; then
    ENFORCE_NOACCEL=0
fi

# Check if the platform is supported (macOS or Linux)
if [[ "$platform" == "Darwin" || "$platform" == "Linux" ]]; then
    echo "Platform: $platform, Architecture: $arch"
else
    echo "Unsupported platform: $platform"
    exit 1
fi

# Function to check and install dependencies for Linux
install_dependencies_linux() {
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        # Install required packages using apt-get
        sudo apt-get update
        sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev
    elif command -v dnf &> /dev/null; then
        # Install required packages using dnf (Fedora)
        sudo dnf install -y gcc-c++ cmake openblas-devel lapack-devel
    elif command -v yum &> /dev/null; then
        # Install required packages using yum (CentOS)
        sudo yum install -y gcc-c++ cmake openblas-devel lapack-devel
    elif command -v zypper &> /dev/null; then
        # Install required packages using zypper (openSUSE)
        sudo zypper install -y gcc-c++ cmake openblas-devel lapack-devel
    elif command -v swupd &> /dev/null; then
        sudo swupd bundle-add c-basic
    else
        echo "Unsupported package manager or unable to install dependencies. were going to ignore it for now."
        #exit 1
    fi
}

# Function to check and install dependencies for macOS
install_dependencies_macos() {
    # Check if Xcode command line tools are installed
    if ! command -v xcode-select &> /dev/null; then
        echo "Xcode command line tools not found. Please install Xcode and try again."
        exit 1
    fi
}

detect_cuda() {
    if [ $ENFORCE_NOACCEL != "1" ]; then
    if command -v nvcc &> /dev/null ; then
        echo "cuda"
    else
        echo "no_cuda"
    fi
    else
        echo "no_cuda"
    fi
}

detect_opencl() {
    # Detect platform
    platform=$(uname -s)

    # Check if OpenCL is available
    if [ $ENFORCE_NOACCEL != "1" ]; then
        if [[ "$platform" == "Linux" ]]; then
            # Check if the OpenCL headers are installed on Linux
            if [ -e "/usr/include/CL/cl.h" ]; then
                echo "opencl"
            else
                echo "no_opencl"
            fi
        elif [[ "$platform" == "Darwin" ]]; then
            # Check if the OpenCL framework is present on macOS
            if [ -e "/System/Library/Frameworks/OpenCL.framework" ]; then
                echo "opencl"
            else
                echo "no_opencl"
            fi
        else
            echo "unsupported"
        fi
    else
        echo "no_opencl"
    fi
}


detect_metal() {
    # Check if the Metal framework is present on macOS
    if [ $ENFORCE_NOACCEL != "1" ]; then
        if [ -e "/System/Library/Frameworks/Metal.framework" ]; then
            echo "metal"
        else
            echo "no_metal"
        fi
    else
        echo "no_metal"
    fi
}
#-------------------------------------------------------
#chat binary building
# Function to build and install LLaMa
build_llama() {
    # Clone submodule and update
    git submodule update --init --recursive
    
    # Change directory to llama.cpp
    cd usr/vendor/llama.cpp || exit 1

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    # Install dependencies based on the platform
    if [[ "$platform" == "Linux" ]]; then
        install_dependencies_linux
    elif [[ "$platform" == "Darwin" ]]; then
        install_dependencies_macos
    fi

    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)

    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DLLAMA_CUBLAS=on"
            CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "arm64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "arm64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        elif [[ "$arch" == "arm64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DLLAMA_CUBLAS=on"
            CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        else
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    echo $CMAKE_ARGS $CMAKE_CUDA_FLAGS
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel $(nproc) || { echo "LLaMa compilation failed. See logs for details."; exit 1; }
    pwd
    # Move the binary to ./usr/bin/ and rename it to "chat" or "chat.exe"
    if [[ "$platform" == "Linux" ]]; then
        cp bin/main ${rootdir}/usr/bin/chat
    elif [[ "$platform" == "Darwin" ]]; then
        cp bin/main ${rootdir}/usr/bin/chat
    fi

    # Change directory back to rootdir
    cd "$rootdir" || exit 1
}

# Function to build and install LLaMa
build_ggml_base() {
    echo "Requesting GGML Binary"
    # Clone submodule and update
    git submodule update --init --recursive
    
    # Change directory to llama.cpp
    cd usr/vendor/ggml || exit 1

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    # Install dependencies based on the platform
    if [[ "$platform" == "Linux" ]]; then
        install_dependencies_linux
    elif [[ "$platform" == "Darwin" ]]; then
        install_dependencies_macos
    fi

    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)

    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DGGML_CUBLAS=on"
            CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DGGML_CLBLAST=on"
        elif [[ "$arch" == "arm64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DGGML_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "arm64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DGGML_METAL=on"
        elif [[ "$arch" == "arm64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DGGML_METAL=on"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DGGML_CUBLAS=on"
            CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DGGML_CLBLAST=on"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DGGML_METAL=on"
        else
            CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    make -j $(nproc) ${1} || { echo "GGML based compilation failed. See logs for details."; exit 1; }
    pwd
    # Move the binary to ./usr/bin/ and rename it to "chat" or "chat.exe"
    if [[ "$platform" == "Linux" ]]; then
        cp bin/${1} ${rootdir}/usr/bin/chat
    elif [[ "$platform" == "Darwin" ]]; then
        cp bin/${1} ${rootdir}/usr/bin/chat
    fi

    # Change directory back to rootdir
    cd "$rootdir" || exit 1
}


build_falcon() {
    # Clone submodule and update
    git submodule update --init --recursive
    
    # Change directory to llama.cpp
    cd usr/vendor/ggllm.cpp || exit 1

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    # Install dependencies based on the platform
    if [[ "$platform" == "Linux" ]]; then
        install_dependencies_linux
    elif [[ "$platform" == "Darwin" ]]; then
        install_dependencies_macos
    fi
    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)
    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DLLAMA_CUBLAS=on"
            CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "arm64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "arm64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        elif [[ "$arch" == "arm64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DLLAMA_CUBLAS=on"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        else
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel $(nproc) || { echo "ggllm compilation failed. See logs for details."; exit 1; }
    pwd
    # Move the binary to ./usr/bin/ and rename it to "chat" or "chat.exe"
    if [[ "$platform" == "Linux" ]]; then
        cp bin/main ${rootdir}/usr/bin/chat
    elif [[ "$platform" == "Darwin" ]]; then
        cp bin/main ${rootdir}/usr/bin/chat
    fi

    # Change directory back to rootdir
    cd "$rootdir" || exit 1
}

#----------------------------------------------------------------

#select working mode
if [ "${1}" == "llama" ]; then
    # Check if LLaMa is already compiled
    if [[ -f ./usr/vendor/llama.cpp/build/bin/main  || -f ./usr/vendor/llama.cpp/build/bin/main.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/llama.cpp/build/bin/${1} ]; then
        cp ./usr/vendor/llama.cpp/build/bin/main ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/llama.cpp/build/bin/main.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "LLaMa binary not found. Building LLaMa..."
        build_llama
    fi
fi

if [ "${1}" == "mpt" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "dolly-v2" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi


if [ "${1}" == "gpt-2" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "gpt-j" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "gpt-neox" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "mnist" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "replit" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "starcoder" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "whisper" ]; then
    # Check if ggml Binary already compiled
    echo "Requested Universal GGML Binary Mode"
    if [[ -f ./usr/vendor/ggml/build/bin/${1}  || -f ./usr/vendor/ggml/build/bin/${1}.exe ]]; then
        echo "${1} binary already compiled. Moving it to ./usr/bin/..."
        if [ -f ./usr/vendor/ggml/build/bin/${1} ]; then
        cp ./usr/vendor/ggml/build/bin/${1} ./usr/bin/chat
        fi
         if [ -f ./usr/vendor/ggml/build/bin/${1}.exe ]; then
        cp ./usr/vendor/ggml/build/bin/${1}.exe ./usr/bin/chat.exe
        fi
    else
        # LLaMa not compiled, build it
        echo "mpt binary not found. Building mpt..."
        build_ggml_base "${1}"
    fi
fi

if [ "${1}" == "falcon" ]; then
    # Check if Falcon ggllm.cpp compiled
    if [[ -f ./vendor/llama.cpp/build/bin/main || -f ./vendor/llama.cpp/build/bin/main.exe ]]; then
        echo "falcon binary already compiled. Moving it to ./usr/bin/..."
        cp ./vendor/ggllm.cpp/build/bin/main ./usr/bin/chat
        cp ./vendor/ggllm.cpp/build/bin/main.exe ./usr/bin/chat.exe
    else
        # LLaMa not compiled, build it
        echo "Falcon Not found! Compiling"
        build_falcon
    fi
fi


# Change directory to ./usr and install npm dependencies
cd ./usr || exit 1

# Function to install dependencies for Linux
install_dependencies_linux() {
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        # Install required packages using apt-get
        sudo apt-get update
        sudo apt-get install -y nodejs npm
    elif command -v dnf &> /dev/null; then
        # Install required packages using dnf (Fedora)
        sudo dnf install -y nodejs npm
    elif command -v yum &> /dev/null; then
        # Install required packages using yum (CentOS)
        sudo yum install -y nodejs npm
    elif command -v zypper &> /dev/null; then
        # Install required packages using zypper (openSUSE)
        sudo zypper install -y nodejs npm
    elif command -v swupd &> /dev/null; then
        sudo swupd bundle-add nodejs-basic
    else
        echo "Unsupported package manager or unable to install dependencies. Exiting."
        exit 1
    fi
}

# Function to install dependencies for macOS
install_dependencies_macos() {
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew and try again."
        exit 1
    fi
}

# Install npm dependencie
if [ -z "$(command -v npm)" ]; then
    if [[ "$platform" == "Linux" ]]; then
        install_dependencies_linux
    elif [[ "$platform" == "Darwin" ]]; then
        install_dependencies_macos
    fi
fi

# Install npm dependencies
npm install --save-dev

# Execute npm run based on the platform and architecture
if [[ "$platform" == "Linux" ]]; then
    if [[ "$arch" == "x86_64" ]]; then
        npm run linux-x64 || { echo "LLaMa npm run linux-x64 failed. See logs for details."; exit 1; }
    elif [[ "$arch" == "aarch64" ]]; then
        npm run linux-arm64 || { echo "LLaMa npm run linux-arm64 failed. See logs for details."; exit 1; }
    fi
elif [[ "$platform" == "Darwin" ]]; then
    if [[ "$arch" == "x86_64" ]]; then
        npm run mac-x64 || { echo "LLaMa npm run mac-x64 failed. See logs for details."; exit 1; }
    elif [[ "$arch" == "arm64" ]]; then
        npm run mac-arm64 || { echo "LLaMa npm run mac-arm64 failed. See logs for details."; exit 1; }
    fi
fi

echo "LLaMa installation completed successfully."

cd ${rootdir}/usr/release-builds/*Zephyrine*/
chmod +x Project*Zephyrine*
"./Project Zephyrine"
