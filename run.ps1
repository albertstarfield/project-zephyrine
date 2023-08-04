$ErrorActionPreference = "Stop"

# Initialize Variables and check the platform and architecture
$platform = $env:OS
$arch = $env:PROCESSOR_ARCHITECTURE

# Save the current working directory to "rootdir" variable (compensate spaces)
$rootdir = Get-Location

$ENFORCE_NOACCEL = 0
if ($env:ENFORCE_NOACCEL) {
    $ENFORCE_NOACCEL = $env:ENFORCE_NOACCEL
}

# Check if the platform is supported (macOS or Linux)
if ($platform -eq "Darwin" -or $platform -eq "Linux") {
    Write-Output "Platform: $platform, Architecture: $arch"
}
else {
    Write-Output "Unsupported platform: $platform"
    exit 1
}

# Function to check and install dependencies for Linux
function Install-Dependencies-Linux {
    # Check and install package manager (winget)
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Output "Installing winget..."
        Invoke-WebRequest -Uri "https://github.com/microsoft/winget-cli/releases/latest/download/winget-cli-x64.msi" -OutFile "winget-cli-x64.msi"
        Start-Process -Wait -FilePath msiexec.exe -ArgumentList "/i winget-cli-x64.msi /quiet"
        Remove-Item -Path "winget-cli-x64.msi" -Force
    }

    # Install required packages using winget
    Write-Output "Installing required packages..."
    winget install --id "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" --exact
    winget install --id "Microsoft.VisualStudio.Component.VC.Tools.ARM64" --exact
    winget install --id "OpenBLAS.OpenBLAS" --exact
}

# Function to check and install dependencies for macOS
function Install-Dependencies-MacOS {
    # Check if Chocolatey is installed
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        Write-Output "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    }

    # Install required packages using Chocolatey
    Write-Output "Installing required packages..."
    choco install -y vcbuildtools
    choco install -y openblas
}

# Function to check and install dependencies based on the platform
function Install-Dependencies {
    if ($platform -eq "Linux") {
        Install-Dependencies-Linux
    }
    elseif ($platform -eq "Darwin") {
        Install-Dependencies-MacOS
    }
}

# Function to detect CUDA on Windows
function Detect-CUDA-Windows {
    if ($ENFORCE_NOACCEL -ne "1") {
        if (Get-Command nvcc -ErrorAction SilentlyContinue) {
            Write-Output "cuda"
        }
        else {
            Write-Output "no_cuda"
        }
    }
    else {
        Write-Output "no_cuda"
    }
}

# Function to detect OpenCL on Windows
function Detect-OpenCL-Windows {
    if ($ENFORCE_NOACCEL -ne "1") {
        if (Test-Path "$env:SystemRoot\System32\OpenCL.dll") {
            Write-Output "opencl"
        }
        else {
            Write-Output "no_opencl"
        }
    }
    else {
        Write-Output "no_opencl"
    }
}

# Function to detect Metal on macOS
function Detect-Metal-MacOS {
    if ($ENFORCE_NOACCEL -ne "1") {
        if (Test-Path "/System/Library/Frameworks/Metal.framework") {
            Write-Output "metal"
        }
        else {
            Write-Output "no_metal"
        }
    }
    else {
        Write-Output "no_metal"
    }
}

# Function to detect acceleration types
function Detect-Acceleration {
    if ($platform -eq "Darwin") {
        $metal = Detect-Metal-MacOS
        if ($metal -eq "metal") {
            Write-Output "metal"
            return
        }
    }

    if ($platform -eq "Windows") {
        $cuda = Detect-CUDA-Windows
        if ($cuda -eq "cuda") {
            Write-Output "cuda"
            return
        }

        $opencl = Detect-OpenCL-Windows
        if ($opencl -eq "opencl") {
            Write-Output "opencl"
            return
        }
    }

    Write-Output "no_accel"
}

# Function to build and install LLaMa
function Build-LLaMa {
    # Clone submodule and update
    git submodule update --init --recursive
    
    # Change directory to llama.cpp
    Set-Location usr/vendor/llama.cpp

    # Create build directory and change directory to it
    New-Item -ItemType Directory -Force -Path "build" | Out-Null
    Set-Location build

    # Install dependencies based on the platform
    Install-Dependencies

    $accelerationType = Detect-Acceleration
    if ($platform -eq "Linux") {
        if ($accelerationType -eq "cuda") {
            $CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
            $CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        }
        elseif ($accelerationType -eq "opencl") {
            $CMAKE_ARGS = "-DLLAMA_CLBLAST=on"
        }
        elseif ($accelerationType -eq "no_accel") {
            $CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        }
    }
    elseif ($platform -eq "Darwin") {
        if ($accelerationType -eq "cuda") {
            $CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
            $CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        }
        elseif ($accelerationType -eq "opencl") {
            $CMAKE_ARGS = "-DLLAMA_CLBLAST=on"
        }
        elseif ($accelerationType -eq "metal") {
            $CMAKE_ARGS = "-DLLAMA_METAL=on"
        }
        elseif ($accelerationType -eq "no_accel") {
            $CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        }
    }

    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel $(Get-WmiObject Win32_ComputerSystem).NumberOfLogicalProcessors | Out-Null
    Set-Location $rootdir

    # Rename the binary to "chat" or "chat.exe"
    if ($platform -eq "Linux" -or $platform -eq "Darwin") {
        Move-Item -Path "usr/vendor/llama.cpp/build/bin/main" -Destination "usr/bin/chat" -Force
    }
}

# Function to build and install ggml base
function Build-GGML-Base {
    param (
        [string]$buildTarget
    )

    # Clone submodule and update
    git submodule update --init --recursive

    # Change directory to ggml
    Set-Location usr/vendor/ggml

    # Create build directory and change directory to it
    New-Item -ItemType Directory -Force -Path "build" | Out-Null
    Set-Location build

    # Install dependencies based on the platform
    Install-Dependencies

    $accelerationType = Detect-Acceleration
    if ($platform -eq "Linux") {
        if ($accelerationType -eq "cuda") {
            $CMAKE_ARGS = "-DGGML_CUBLAS=on"
            $CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        }
        elseif ($accelerationType -eq "opencl") {
            $CMAKE_ARGS = "-DGGML_CLBLAST=on"
        }
        elseif ($accelerationType -eq "no_accel") {
            $CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        }
    }
    elseif ($platform -eq "Darwin") {
        if ($accelerationType -eq "cuda") {
            $CMAKE_ARGS = "-DGGML_CUBLAS=on"
            $CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        }
        elseif ($accelerationType -eq "opencl") {
            $CMAKE_ARGS = "-DGGML_CLBLAST=on"
        }
        elseif ($accelerationType -eq "metal") {
            $CMAKE_ARGS = "-DGGML_METAL=on"
        }
        elseif ($accelerationType -eq "no_accel") {
            $CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        }
    }

    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel $(Get-WmiObject Win32_ComputerSystem).NumberOfLogicalProcessors | Out-Null
    Set-Location $rootdir

    # Rename the binary to "chat" or "chat.exe"
    if ($platform -eq "Linux" -or $platform -eq "Darwin") {
        Move-Item -Path "usr/vendor/ggml/build/bin/$buildTarget" -Destination "usr/bin/chat" -Force
    }
}

# Function to build and install Falcon
function Build-Falcon {
    # Clone submodule and update
    git submodule update --init --recursive

    # Change directory to ggllm.cpp
    Set-Location usr/vendor/ggllm.cpp

    # Create build directory and change directory to it
    New-Item -ItemType Directory -Force -Path "build" | Out-Null
    Set-Location build

    # Install dependencies based on the platform
    Install-Dependencies

    $accelerationType = Detect-Acceleration
    if ($platform -eq "Linux") {
        if ($accelerationType -eq "cuda") {
            $CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
            $CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        }
        elseif ($accelerationType -eq "opencl") {
            $CMAKE_ARGS = "-DLLAMA_CLBLAST=on"
        }
        elseif ($accelerationType -eq "no_accel") {
            $CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        }
    }
    elseif ($platform -eq "Darwin") {
        if ($accelerationType -eq "cuda") {
            $CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
        }
        elseif ($accelerationType -eq "opencl") {
            $CMAKE_ARGS = "-DLLAMA_CLBLAST=on"
        }
        elseif ($accelerationType -eq "metal") {
            $CMAKE_ARGS = "-DLLAMA_METAL=on"
        }
        elseif ($accelerationType -eq "no_accel") {
            $CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        }
    }

    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel $(Get-WmiObject Win32_ComputerSystem).NumberOfLogicalProcessors | Out-Null
    Set-Location $rootdir

    # Rename the binary to "chat" or "chat.exe"
    if ($platform -eq "Linux" -or $platform -eq "Darwin") {
        Move-Item -Path "usr/vendor/ggllm.cpp/build/bin/main" -Destination "usr/bin/chat" -Force
    }
}

function Install-Dependencies {
    # Install Chocolatey if not installed
    if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    }

    # Install Node.js and npm using Chocolatey
    if (!(Get-Command node -ErrorAction SilentlyContinue) -or !(Get-Command npm -ErrorAction SilentlyContinue)) {
        choco install nodejs npm -y
    }
}

function Detect-Acceleration {
    # TODO: Implement detection of CUDA, OpenCL, and Metal on Windows
    return "no_accel"
}

# Initialize Variables and check the platform and architecture
$platform = (Get-CimInstance Win32_OperatingSystem).Caption
$arch = (Get-CimInstance Win32_Processor).AddressWidth
$rootdir = Get-Location
$accelerationType = "no_accel"

# Check if the platform is supported (Windows 10 or 11)
if ($platform -match "Windows 10|Windows 11") {
    Write-Host "Platform: $platform, Architecture: $arch"
}
else {
    Write-Host "Unsupported platform: $platform"
    exit 1
}

# Change directory to ./usr and install npm dependencies
Set-Location ./usr

# Install npm dependencies
Install-Dependencies

# Install npm dependencies
npm install --save-dev

# Execute npm run based on the platform and architecture
if ($platform -match "Windows 10" -and $arch -eq "64-bit") {
    npm run windows-x64
}
elseif ($platform -match "Windows 11" -and $arch -eq "64-bit") {
    npm run windows-x64
}
elseif ($platform -match "Windows 11" -and $arch -eq "32-bit") {
    npm run windows-x86
}
else {
    Write-Host "Unsupported platform or architecture: $platform, $arch"
    exit 1
}

# Change directory back to rootdir
Set-Location $rootdir

#select working mode
if ($args[0] -eq "llama") {
    # Check if LLaMa is already compiled
    if (Test-Path "usr/vendor/llama.cpp/build/bin/main" -or Test-Path "usr/vendor/llama.cpp/build/bin/main.exe") {
        Write-Host "LLaMa binary already compiled. Moving it to ./usr/bin/..."
        Move-Item -Path "usr/vendor/llama.cpp/build/bin/main" -Destination "usr/bin/chat" -Force
    }
    else {
        # LLaMa not compiled, build it
        Write-Host "LLaMa binary not found. Building LLaMa..."
        Build-LLaMa
    }
}

if ($args[0] -eq "mpt") {
    # Check if ggml Binary already compiled
    Write-Host "Requested Universal GGML Binary Mode"
    if (Test-Path "usr/vendor/ggml/build/bin/mpt" -or Test-Path "usr/vendor/ggml/build/bin/mpt.exe") {
        Write-Host "LLaMa binary already compiled. Moving it to ./usr/bin/..."
        Move-Item -Path "usr/vendor/ggml/build/bin/mpt" -Destination "usr/bin/mpt" -Force
    }
    else {
        # LLaMa not compiled, build it
        Write-Host "mpt binary not found. Building mpt..."
        Build-GGML-Base "mpt"
    }
}

if ($args[0] -eq "dolly-v2") {
    # Check if ggml Binary already compiled
    Write-Host "Requested Universal GGML Binary Mode"
    if (Test-Path "usr/vendor/ggml/build/bin/mpt" -or Test-Path "usr/vendor/ggml/build/bin/mpt.exe") {
        Write-Host "LLaMa binary already compiled. Moving it to ./usr/bin/..."
        Move-Item -Path "usr/vendor/ggml/build/bin/mpt" -Destination "usr/bin/mpt" -Force
    }
    else {
        # LLaMa not compiled, build it
        Write-Host "mpt binary not found. Building mpt..."
        Build-GGML-Base "mpt"
    }
}

if ($args[0] -eq "gpt-2") {
    # Check if ggml Binary already compiled
    Write-Host "Requested Universal GGML Binary Mode"
    if (Test-Path "usr/vendor/ggml/build/bin/mpt" -or Test-Path "usr/vendor/ggml/build/bin/mpt.exe") {
        Write-Host "LLaMa binary already compiled. Moving it to ./usr/bin/..."
        Move-Item -Path "usr/vendor/ggml/build/bin/mpt" -Destination "usr/bin/mpt" -Force
    }
    else {
        # LLaMa not compiled, build it
        Write-Host "mpt binary not found. Building mpt..."
        Build-GGML-Base "mpt"
    }
}

Write-Host "Installation and compilation completed successfully!"
