#!/bin/sh

# Check if we're running in an older "sh" environment
if [ "${0##*/}" = "sh" ]; then
# Relaunch the script using bash to get better functionality
exec /bin/bash "$0" "$@"
fi
#------
# Do not let the user use root to run this, unless it chroot which in that case the user is advanced enough to know what they're up to
if [ -z "${SU_BYPASS_CHECK+set}" ] || [ ${SU_BYPASS_CHECK} != 1 ]; then
  if [ $EUID -eq 0 ]; then
    echo "Do not run this script as superuser unless you know what you're doing!"
    exit 1
  fi
fi

#-------
#For some reason if you code it #!/bin/bash and your are defaulting to /usr/bin/fish doing  that built using x86_64 and enforce zsh with arch -arm64 the installation will be mixed with x86_64 binary and cause havoc on the module installation 
set -e

# Initialize Variables and check the platform and architecture
platform=$(uname -s)
arch=$(uname -m)
if [[ -n $(uname) && "$(uname)" == "Darwin" ]]; then
allocThreads=$(($(sysctl -n hw.logicalcpu 2> /dev/null | sed 's/.*: //') / 4))
elif [[ -n $(uname) && "$(uname)" == "CYGWINNT" ]]; then   # Windows (in Bash)
allocThreads=$(( $(wmic cpu get/Cores 2> /dev/null | sed 's/.*: //') / 4))
elif [[ -n $(uname) && "$(uname)" == "MSYS2" ]]; then   # MSYS2 (Windows)
allocThreads=$(($(wmic cpu get/Cores 2> /dev/null | sed 's/.*: //') / 4))
else
allocThreads=$(($(nproc --all) / 4))
fi
#Correction if threads allocation gone wrong!
if [ ${allocThreads} -lt 1 ]; then
allocThreads=1
fi

# Save the current working directory to "rootdir" variable (compensate spaces)
rootdir="$(pwd)"
export CONDA_PREFIX="${rootdir}/conda_python_modules"
export LC_CTYPE=UTF-8
export N_PREFIX="${rootdir}/usr/nodeLocalRuntime" #custom PREFIX location for this specific adelaide installation
# allow to prioritize N_PREFIX and CONDA_PREFIX binary over global
export PATH="${N_PREFIX}/bin:${CONDA_PREFIX}/bin:${PATH}"
echo $PATH


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
        sudo apt-get install -y build-essential python3 cmake libopenblas-dev liblapack-dev
    elif command -v dnf &> /dev/null; then
        # Install required packages using dnf (Fedora)
        sudo dnf install -y gcc-c++ cmake openblas-devel python lapack-devel
    elif command -v yum &> /dev/null; then
        # Install required packages using yum (CentOS)
        sudo yum install -y gcc-c++ cmake openblas-devel python lapack-devel
    elif command -v zypper &> /dev/null; then
        # Install required packages using zypper (openSUSE)
        sudo zypper install -y gcc-c++ cmake openblas-devel python lapack-devel
    elif command -v pacman &> /dev/null; then
        # Install required packages using pacman (Arch Linux, Manjaro, etc.)
        sudo pacman -Syu --needed base-devel python cmake openblas lapack
    elif command -v swupd &> /dev/null; then
        # Install required packages using swupd (SUSE)
        sudo swupd bundle-add c-basic
    else
        echo "Unsupported package manager or unable to install dependencies. We're going to ignore it for now."
        # exit 1
    fi
}

install_dependencies_windows() {
    # Install required packages using pacman (MSYS2)
    pacman  -Syyu --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-python3 mingw-w64-x86_64-openblas mingw-w64-x86_64-lapack
}

# Function to check and install dependencies for macOS
install_dependencies_macos() {
    set +e
    echo "macOS/darwin based system detected!"
    # Check if Xcode command line tools are installed
    if ! command -v xcode-select &> /dev/null; then
        echo "Xcode command line tools not found. Please install Xcode and try again."
        echo "Kindly be aware of the necessity to install the compatible Xcode version corresponding to the specific macOS iteration. For instance, macOS Sonoma mandates the utilization of Xcode 15 Beta, along with its corresponding Xcode Command Line Tools version 15."
        exit 1
    else
        xcode-select --install
        sudo xcodebuild -license accept
    fi
    #if [ ! -d /Applications/Xcode.app ]; then
    #    echo "XCode.app main app isn't install, Please install and try again"
    #fi
    if ! command -v brew &> /dev/null; then
        echo "Homebrew command line tools not found. Please install Homebrew and try again."
        
        exit 1
    else 

    #https://stackoverflow.com/questions/64963370/error-cannot-install-in-homebrew-on-arm-processor-in-intel-default-prefix-usr
    # Be aware apple slick may have conflict with x86_64 homebrew package and cause this to fail

    sudo chown -R "$USER":admin /opt/homebrew/ #arm64 darwin
    #/usr/local/homebrew/
    sudo chown -R "$USER":admin /usr/local/homebrew/ #x86_64 darwin
    brew doctor
    #rm -rf "/opt/homebrew/Library/Taps/homebrew/homebrew-core"
    brew tap homebrew/core
    brew tap apple/apple http://github.com/apple/homebrew-apple
    brew upgrade --greed
    brew install python node cmake mas
    echo "Installing XCode..."
    mas search XCode
    mas install 497799835 #XCode appid from mas search XCode
    

    fi
    echo "Upgrading to the recommended node Version!"
    set -e
    node -v
    npm -g install n
    #n 20.11.1 #They removed it!
    n latest #To prevent this problem we might grab LTS or latest instead
    echo "did it work?"
    node -v
    
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


# Clone submodule repositories
# Function to clone submodule
clone_submodule() {
    local path="$1"
    local url="$2"
    local commit="$3"
    echo "Cloning submodule: $path from $url"
    git clone --recurse-submodules --single-branch --branch "$commit" "$url" "$path"
    #git clone --branch "$commit" "$url" "$path"

}


importsubModuleManually(){
    #clone_submodule "${rootdir}/usr/vendor/llama.cpp" "https://github.com/ggerganov/llama.cpp" "93356bd"
    clone_submodule "${rootdir}/usr/vendor/llama.cpp" "https://github.com/ggerganov/llama.cpp" "master"
    clone_submodule "${rootdir}/usr/vendor/ggllm.cpp" "https://github.com/cmp-nct/ggllm.cpp" "master"
    clone_submodule "${rootdir}/usr/vendor/ggml" "https://github.com/ggerganov/ggml" "master"
    clone_submodule "${rootdir}/usr/vendor/llama-gguf.cpp" "https://github.com/ggerganov/llama.cpp" "master"
    clone_submodule "${rootdir}/usr/vendor/whisper.cpp" "https://github.com/ggerganov/whisper.cpp" "master"
    clone_submodule "${rootdir}/usr/vendor/gemma.cpp" "https://github.com/google/gemma.cpp" "main"
    # Screw git submodule and .gitmodules system, its useless, crap, and ignore all the listing and only focused llama.cpp as always and ignore everything else
}


cleanInstalledFolder(){
    set +e
    echo "Cleaning Installed Folder to lower the chance of interfering with the installation process"
    npm cache clean --force
    rm -rf ${rootdir}/usr/vendor ${rootdir}/usr/node_modules ${CONDA_PREFIX} ${N_PREFIX}
    mkdir ${rootdir}/usr/vendor
    echo "Should be done"
    set -e
}
build_llama() {
    # Clone submodule and   update
    
    # Change directory to llama.cpp
    cd usr/vendor/llama.cpp || exit 1
    git checkout 93356bd #return to ggml era not the dependencies breaking gguf model mode 

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)

    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        else
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        fi

        if [ "$(uname -m)" == "arm64" ]; then
            echo "Enforcing compilation to $(uname -m), Probably cmake wont listen!"
            export CMAKE_HOST_SYSTEM_PROCESSOR="arm64"
            ENFORCE_ARCH_COMPILATION="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        else
            ENFORCE_ARCH_COMPILATION=""
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    echo $CMAKE_ARGS $CMAKE_CUDA_FLAGS
    set -x
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS $ENFORCE_ARCH_COMPILATION
    set +x

    # Build with multiple cores
    echo "This is the architecture $(uname -m) unless the cmake becoming asshole and detect arm64 as x86_64"
    cmake --build . --config Release --parallel ${allocThreads} || { echo "LLaMa compilation failed. See logs for details."; exit 1; }
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
    
    
    # Change directory to llama.cpp
    cd usr/vendor/llama-gguf.cpp || exit 1

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)

    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ ( "$arch" == "arm64" || "$arch" == "aarch64" ) && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        else
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        fi
        if [ "$(uname -m)" == "arm64" ]; then
            echo "Enforcing compilation to $(uname -m), Probably cmake wont listen!"
            export CMAKE_HOST_SYSTEM_PROCESSOR="arm64"
            ENFORCE_ARCH_COMPILATION="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        else
            ENFORCE_ARCH_COMPILATION=""
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    echo $CMAKE_ARGS $CMAKE_CUDA_FLAGS
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel ${allocThreads} || { echo "LLaMa compilation failed. See logs for details."; exit 1; }
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

# Function to build and install ggml
build_ggml_base() {
    echo "Requesting GGML Binary"

    
    # Change directory to llama.cpp
    cd usr/vendor/ggml || exit 1

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)

    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_METAL=on"
        elif [[ "$arch" == "arm64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_METAL=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CLBLAST=on"
        elif [[ "$arch" == "arm64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "arm64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_METAL=on"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CLBLAST=on"
        else
            CMAKE_ARGS="${CMAKE_ARGS} -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        fi
        if [ "$(uname -m)" == "arm64" ]; then
            echo "Enforcing compilation to $(uname -m), Probably cmake wont listen!"
            export CMAKE_HOST_SYSTEM_PROCESSOR="arm64"
            ENFORCE_ARCH_COMPILATION="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        else
            ENFORCE_ARCH_COMPILATION=""
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    make -j ${allocThreads} ${1} || { echo "GGML based compilation failed. See logs for details."; exit 1; }
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


# Function to build and install ggml
build_gemma_base() {
    echo "Requesting Google Gemma Binary"

    
    # Change directory to llama.cpp
    cd usr/vendor/gemma.cpp || exit 1

    # Create build directory and change directory to it
    if [ ! -d build ]; then
    mkdir -p build
    fi

    cd build || exit 1
    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)

    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS=""
            echo "Gemma is CPU SIMD only!"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS=""
            echo "Gemma is CPU SIMD only!"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS=""
            echo "Gemma is CPU SIMD only!"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS=""
            echo "Gemma is CPU SIMD only!"
        else
            CMAKE_ARGS=""
            echo "Gemma is CPU SIMD only!"
        fi
        if [ "$(uname -m)" == "arm64" ]; then
            echo "Enforcing compilation to $(uname -m), Probably cmake wont listen!"
            export CMAKE_HOST_SYSTEM_PROCESSOR="arm64"
            ENFORCE_ARCH_COMPILATION="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        else
            ENFORCE_ARCH_COMPILATION=""
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    make -j ${allocThreads} ${1} || { echo "Gemma compilation failed. See logs for details."; exit 1; }
    pwd
    # Change directory back to rootdir
    cd "$rootdir" || exit 1
}


build_falcon() {
    # Change directory to llama.cpp
    cd usr/vendor/ggllm.cpp || exit 1

    # Create build directory and change directory to it
    mkdir -p build
    cd build || exit 1

    cuda=$(detect_cuda)
    opencl=$(detect_opencl)
    metal=$(detect_metal)
    if [[ "$platform" == "Linux" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CUBLAS=on"
            #CMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
        elif [[ "$arch" == "amd64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$arch" == "arm64" && "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "arm64" && "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        elif [[ "$arch" == "amd64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        elif [[ "$arch" == "arm64" && "$opencl" == "no_opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        else
            echo "No special Acceleration, Ignoring"
        fi
    elif [[ "$platform" == "Darwin" ]]; then
        if [[ "$cuda" == "cuda" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CUBLAS=on"
        elif [[ "$metal" == "metal" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_METAL=on"
        elif [[ "$opencl" == "opencl" ]]; then
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_CLBLAST=on"
        else
            CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        fi
        if [ "$(uname -m)" == "arm64" ]; then
            echo "Enforcing compilation to $(uname -m), Probably cmake wont listen!"
            export CMAKE_HOST_SYSTEM_PROCESSOR="arm64"
            ENFORCE_ARCH_COMPILATION="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        else
            ENFORCE_ARCH_COMPILATION=""
        fi
    else
        echo "Unsupported platform: $platform"
        exit 1
    fi
    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel ${allocThreads} || { echo "ggllm compilation failed. See logs for details."; exit 1; }
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

    #add Kernel support 1_windows

    if [ "${platform}" == "Darwin" ]; then
        targetFolderPlatform="0_macOS"
    fi

    if [ "${platform}" == "Linux" ]; then
        targetFolderPlatform="2_Linux"
    fi




    # since the cuda binaries or the opencl binaries can be used as the nonaccel binaries to we can just copy the same binaries to the folder
    # This naming system was introduced due to the Windows different LLMBackend precompiled versions (check llama.cpp and ggllm.cpp release tabs and see the different version of version)
    # example directory ./usr/bin/0_macOS/arm64/LLMBackend-llama-noaccel

    echo "Cleaning binaries Replacing with new ones"
    rm -rf "${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}"
    if [ ! -d "${rootdir}/usr/bin" ]; then
        mkdir "${rootdir}/usr/bin"
        mkdir "${rootdir}/usr/bin/${targetFolderPlatform}"
    fi
    if [ ! -d "${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}" ]; then
        mkdir "${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}"
        echo "${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}"
    fi
    


    #You know what lets abandon Windows enforced binary structuring and find another way on how to execute other way to have specific acceleration on Windows
    cd ${rootdir}
    build_llama
    cd ${rootdir}
    
    mkdir ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/llama
    cp ./usr/vendor/llama.cpp/build/bin/main ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/llama/LLMBackend-llama
    echo "Copying any Acceleration and Debugging Dependencies for LLaMa GGML v2 v3 Legacy"
    cp -r ./usr/vendor/llama.cpp/build/bin/* ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/llama

    cd ${rootdir}
    build_llama_gguf
    cd ${rootdir}
    mkdir ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/llama-gguf
    cp ./usr/vendor/llama-gguf.cpp/build/bin/main ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/llama-gguf/LLMBackend-llama-gguf
    echo "Copying any Acceleration and Debugging Dependencies for LLaMa GGUF Neo Model"
    cp -r ./usr/vendor/llama-gguf.cpp/build/bin/* ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/llama-gguf

    cd ${rootdir}
    build_falcon
    cd ${rootdir}

    mkdir ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/falcon
    cp ./usr/vendor/ggllm.cpp/build/bin/main ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/falcon/LLMBackend-falcon
    echo "Copying any Acceleration and Debugging Dependencies for Falcon"
    cp -r ./usr/vendor/ggllm.cpp/build/bin/* ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/falcon/
    
    cd ${rootdir}
    build_ggml_base gpt-j
    cd ${rootdir}

    #./usr/vendor/ggml/build/bin/${1} location of the compiled binary ggml based
    mkdir ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/ggml-gptj
    echo "Copying any Acceleration and Debugging Dependencies for gpt-j"
    cp -r ./usr/vendor/ggml/build/bin/* ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/ggml-gptj
    cp ./usr/vendor/ggml/build/bin/gpt-j ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/ggml-gptj/LLMBackend-gpt-j

    cd ${rootdir}
    #gemma
    build_gemma_base
    cd ${rootdir}

    #./usr/vendor/ggml/build/bin/${1} location of the compiled binary ggml based
    mkdir ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/googleGemma
    echo "Copying any Acceleration and Debugging Dependencies for googleGemma"
    cp -r ./usr/vendor/gemma.cpp/build/* ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/googleGemma
    cp ./usr/vendor/gemma.cpp/build/gemma ${rootdir}/usr/bin/${targetFolderPlatform}/${targetFolderArch}/googleGemma/LLMBackend-gemma

    cd ${rootdir}
}


#buildLLMBackend

fix_permisssion_universal(){
    set +e
    echo "Fixing Universal issue"
    sudo chmod -R 777 ${CONDA_PREFIX} ${N_PREFIX}
    set -e
}



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
    npm -g install n
    #n 20.11.1 #They removed it!
    n latest #To prevent this problem we might grab LTS or latest instead
    node -v
}


EnforcingDependencies(){
# Install npm dependencie
if [[ "$platform" == "Linux" ]]; then
    install_dependencies_linux
elif [[ "$platform" == "Darwin" ]]; then
    install_dependencies_macos
elif [[ "$platform" == "MSYS2" ]]; then
    install_dependencies_windows
fi
}

# Install npm dependencies
if [[ ! -f ${rootdir}/installed.flag || "${FORCE_REBUILD}" == "1" ]]; then
    cleanInstalledFolder
    echo "Enforcing Check of Dependencies!"
    EnforcingDependencies
    echo "Enforcing latest npm"
    npm install npm@latest
    echo "Installing Modules"
    npm install --save-dev
    npm audit fix
    npx --openssl_fips='' electron-rebuild
    importsubModuleManually
    buildLLMBackend
    fix_permisssion_universal
    touch ${rootdir}/installed.flag
fi
cd "${rootdir}/usr"
node -v
npm -v
npm start
