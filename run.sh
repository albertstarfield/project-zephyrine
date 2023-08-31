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
    set +e
    # Check if Xcode command line tools are installed
    if ! command -v xcode-select &> /dev/null; then
        echo "Xcode command line tools not found. Please install Xcode and try again."
        echo "Kindly be aware of the necessity to install the compatible Xcode version corresponding to the specific macOS iteration. For instance, macOS Sonoma mandates the utilization of Xcode 15 Beta, along with its corresponding Xcode Command Line Tools version 15."
        exit 1
    else
        
        xcode-select --install
        xcodebuild -license accept
    fi
    if ! command -v brew &> /dev/null; then
        echo "Homebrew command line tools not found. Please install Homebrew and try again."
        exit 1
    else 
    brew doctor
    #rm -rf "/opt/homebrew/Library/Taps/homebrew/homebrew-core"
    brew tap homebrew/core
    brew tap apple/apple http://github.com/apple/homebrew-apple
    brew install node
    fi
    echo "Upgrading to the latest Node Version!"
    set -e
    sudo npm install -g n
    sudo n latest
    sudo node -v
    
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
    git checkout 99d29c0094476c4962023036ecd61a3309d0e16b #return to ggml era not the dependencies breaking gguf model mode 

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
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
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

build_llama_gguf() {
    # Clone submodule and update
    git submodule update --init --recursive
    
    # Change directory to llama.cpp
    cd usr/vendor/llama-gguf.cpp || exit 1

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
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$metal" == "metal" ]]; then
            CMAKE_ARGS="-DLLAMA_METAL=on"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="-DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
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
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
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
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
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
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
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


buildLLMBackend(){
    #Compile all binaries with specific version and support
    
    echo "Platform: $platform, Architecture: $arch" 
    #platform darwin and Linux
    #arch arm64 (darwin) x86_64 aarch64 (linux)
    pwd
    
    targetFolderArch="${arch}"
    if [ "${arch}" == "aarch64" ]; then
        targetFolderArch="arm64"
    fi

     if [ "${arch}" == "x86_64" ]; then
        targetFolderArch="x64"
    fi


    if [ "${platform}" == "Darwin" ]; then
        targetFolderPlatform="0_macOS"
    fi

    if [ "${platform}" == "Linux" ]; then
        targetFolderPlatform="2_Linux"
    fi

    # since the cuda binaries or the opencl binaries can be used as the nonaccel binaries to we can just copy the same binaries to the folder
    # This naming system was introduced due to the Windows different LLMBackend precompiled versions (check llama.cpp and ggllm.cpp release tabs and see the different version of version)
    # example directory ./usr/bin/0_macOS/arm64/LLMBackend-llama-noaccel

    
    cd ${rootdir}
    build_llama
    cd ${rootdir}
    cp ./usr/vendor/llama.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-noaccel
    cp ./usr/vendor/llama.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-cuda
    cp ./usr/vendor/llama.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-opencl
    cp ./usr/vendor/llama.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-openblas

    cd ${rootdir}
    build_llama_gguf
    cd ${rootdir}

    cp ./usr/vendor/llama-gguf.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-gguf-noaccel
    cp ./usr/vendor/llama-gguf.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-gguf-cuda
    cp ./usr/vendor/llama-gguf.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-gguf-opencl
    cp ./usr/vendor/llama-gguf.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-llama-gguf-openblas

    cd ${rootdir}
    build_falcon
    cd ${rootdir}

    cp ./usr/vendor/ggllm.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-falcon-noaccel
    cp ./usr/vendor/ggllm.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-falcon-cuda
    cp ./usr/vendor/ggllm.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-falcon-opencl
    cp ./usr/vendor/ggllm.cpp/build/bin/main ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-falcon-openblas
    
    cd ${rootdir}
    build_ggml_base mpt
    build_ggml_base gpt-2
    build_ggml_base gpt-j
    build_ggml_base gpt-neox
    cd ${rootdir}

    #./usr/vendor/ggml/build/bin/${1} location of the compiled binary ggml based
    cp ./usr/vendor/ggml/build/bin/mpt ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-mpt-noaccel
    cp ./usr/vendor/ggml/build/bin/mpt ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-mpt-cuda

    cp ./usr/vendor/ggml/build/bin/gpt-2 ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-gpt-2-noaccel
    cp ./usr/vendor/ggml/build/bin/gpt-2 ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-gpt-2-cuda

    cp ./usr/vendor/ggml/build/bin/gpt-j ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-gpt-j-noaccel
    cp ./usr/vendor/ggml/build/bin/gpt-j ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-gpt-j-cuda

    cp ./usr/vendor/ggml/build/bin/gpt-neox ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-gpt-neox-noaccel
    cp ./usr/vendor/ggml/build/bin/gpt-neox ./usr/bin/${targetFolderPlatform}/${targetFolderArch}/LLMBackend-gpt-neox-cuda

    cd ${rootdir}
}


#buildLLMBackend

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

    echo "Upgrading to the latest Node Version!"
    sudo npm install -g n
    sudo n latest
    sudo node -v
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
if [[ ! -f ${rootdir}/installed.flag || "${FORCE_REBUILD}" == "1" ]]; then
    npm install --save-dev
    buildLLMBackend
    touch ${rootdir}/installed.flag
fi
cd ${rootdir}/usr
node -v
npm -v
npm start
