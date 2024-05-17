# Initialize Variables
$platform = [System.Environment]::OSVersion.Platform
$arch = [System.Environment]::Is64BitOperatingSystem


# Get the current platform
$currentPlatform = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")

# Check if the platform is ARM64
if ($currentPlatform -eq "ARM64") {
    Write-Host "Regrettably, Windows 10+ arm64 is not currently supported. It is important to clarify that this limitation is not due to any deliberate effort to restrict support exclusively to Apple devices or their proprietary arm64 CPUs. Rather, the backend infrastructure currently lacks the capability to compile for non-Apple arm64 architectures, as its focus is primarily on supporting Apple silicon. Efforts are underway to implement workaround solutions, albeit through somewhat unconventional means. An alternative approach involves running the application on a Linux arm64 virtual machine, which is likely to yield favorable results."
    Exit
}

# Check if the script is running with administrative privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# If not running as administrator, relaunch with administrative rights
if (-not $isAdmin) {
    Write-Host "Script is not running as superuser. Relaunching with superuser privileges..."
    Start-Process powershell.exe -Verb RunAs -ArgumentList ("-File `"$PSCommandPath`"") -Wait
    Exit
}

# Save the current working directory to "rootdir" variable (compensate spaces)
$rootdir = Get-Location
$env:CONDA_PREFIX = "$rootdir\conda_python_modules"
$env:LC_CTYPE = "UTF-8"
$env:N_PREFIX = "$rootdir\nodeLocalRuntime" #custom PREFIX location for this specific adelaide installation

# allow to prioritize N_PREFIX and CONDA_PREFIX binary over global
$env:PATH = "$env:N_PREFIX\bin;$env:CONDA_PREFIX\bin;$env:PATH"
Write-Output $env:PATH

# Check if ENFORCE_NOACCEL is defined, if not set it to 0
if (-not $env:ENFORCE_NOACCEL) {
    $env:ENFORCE_NOACCEL = 0
}


#------------------------------------------------------------------

# Check if the platform is supported (Windows 10 22H2) and if winget binary is installed
$osVersion = [System.Environment]::OSVersion.Version
$platform = [System.Environment]::OSVersion.Platform

if ($platform -eq 'Win32NT' -and $osVersion -ge [Version]'10.0.22509') {
    # Check if winget binary is installed
    $wingetPath = Get-Command -Name winget -ErrorAction SilentlyContinue
    if ($wingetPath) {
        Write-Output "Platform: Windows 10 22H2, Architecture: $($env:PROCESSOR_ARCHITECTURE)"
    } else {
        Write-Output "winget is not installed."
        exit 1
    }
} else {
    Write-Output "Unsupported platform or version: $($osVersion.ToString())"
    exit 1
}



#--------------------------------------------------------------------

function Install-DependenciesWindows {
    # Check if Chocolatey is installed, if not, install it
    
    $chocoPath = Get-Command -Name choco -ErrorAction SilentlyContinue
    if (-not $chocoPath) {
        Write-Output "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    }
    

    choco install cmake --installargs 'ADD_CMAKE_TO_PATH=System' -y
    # Install required packages using Chocolatey
    choco install -y python3 git nodejs
    # Now where is build-essential equivalent?
    choco install -y make msys2

    # Check if nmake is installed, if not, install it
    $nmakePath = Get-Command -Name nmake -ErrorAction SilentlyContinue
    

    if (-not $nmakePath) {
        Write-Output "Installing build-essential bundle package but for Windows 10+..."
        Write-Output "Yes, it's possible to do it without seen or manual GUI on Windows"
        Write-Output "Console may stay here for a while until 4-6 GB downloads some necessary dependencies!"
        Invoke-WebRequest -Uri "https://aka.ms/vs/16/release/vs_buildtools.exe" -OutFile vs_buildtools.exe
        #Reference : https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-community?view=vs-2022
        Start-Process -FilePath .\vs_buildtools.exe -ArgumentList "-q --wait --norestart --nocache --installPath C:\BuildTools --add Microsoft.VisualStudio.Component.VC.CoreIde --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.Component.MSBuild --add Microsoft.VisualStudio.Component.NuGet --add Microsoft.Component.MSBuild --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core --add Microsoft.VisualStudio.Component.VC.CoreIde --add Microsoft.VisualStudio.Component.Roslyn.Compiler" -Wait
        Start-Sleep -Seconds 10
        # Define the path to nmake.exe
        $nmakePath = Get-ChildItem -Path "C:\BuildTools\VC\Tools\MSVC\" -Filter "16.*.*" -Directory | Select-Object -ExpandProperty FullName | ForEach-Object { Join-Path -Path $_ -ChildPath "bin\Hostx64\x64" }

        # Add nmake path to the PATH environment variable
        $env:Path += ";$nmakePath"

        # Reload the PATH environment variable
        [Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)

        # Check if nmake is now accessible in the PATH
        nmake
    }



    Import-Module $env:ChocolateyInstall\helpers\chocolateyProfile.psm1
    refreshenv #refreshing environment

    

    echo "Upgrading to the latest Node Version!"
    nvm install 20.11.1
    nvm use 20.11.1
    node -v

}

#---------------------------------------------------------------------

function Detect-CUDA {
    if ($env:ENFORCE_NOACCEL -ne "1") {
        $nvccPath = Get-Command -Name nvcc -ErrorAction SilentlyContinue
        if ($nvccPath) {
            Write-Output "cuda"
        } else {
            Write-Output "no_cuda"
        }
    } else {
        Write-Output "no_cuda"
    }
}

function Detect-OpenCL {
    # Detect platform
    $platform = [System.Environment]::OSVersion.Platform

    # Check if OpenCL is available
    if ($env:ENFORCE_NOACCEL -ne "1") {
        if ($platform -eq 'Win32NT') {
            # Check if OpenCL is installed on Windows
            $openclInstalled = Test-Path "C:\Windows\System32\OpenCL.dll"
            if ($openclInstalled) {
                Write-Output "opencl"
            } else {
                Write-Output "no_opencl"
            }
        } elseif ($platform -eq 'Unix') {
            # Check if the OpenCL headers are installed on Unix-like systems
            $openclHeadersInstalled = Test-Path "/usr/include/CL/cl.h"
            if ($openclHeadersInstalled) {
                Write-Output "opencl"
            } else {
                Write-Output "no_opencl"
            }
        } else {
            Write-Output "unsupported"
        }
    } else {
        Write-Output "no_opencl"
    }
}

#-------------------------------------------------------------------------------------------

function Clone-Submodule {
    param(
        [string]$path,
        [string]$url,
        [string]$commit
    )

    Write-Output "Cloning submodule: $path from $url"
    git clone --recurse-submodules --single-branch --branch $commit $url $path
}

function Import-SubmoduleManually {
    #Clone-Submodule "${rootdir}\usr\vendor\llama.cpp" "https://github.com/ggerganov/llama.cpp" "93356bd"
    Clone-Submodule "${rootdir}\usr\vendor\llama.cpp" "https://github.com/ggerganov/llama.cpp" "master"
    Clone-Submodule "${rootdir}\usr\vendor\ggllm.cpp" "https://github.com/cmp-nct/ggllm.cpp" "master"
    Clone-Submodule "${rootdir}\usr\vendor\ggml" "https://github.com/ggerganov/ggml" "master"
    Clone-Submodule "${rootdir}\usr\vendor\llama-gguf.cpp" "https://github.com/ggerganov/llama.cpp" "master"
    Clone-Submodule "${rootdir}\usr\vendor\whisper.cpp" "https://github.com/ggerganov/whisper.cpp" "master"
    Clone-Submodule "${rootdir}\usr\vendor\gemma.cpp" "https://github.com/google/gemma.cpp" "master"
    # Screw git submodule and .gitmodules system, its useless, crap, and ignore all the listing and only focused llama.cpp as always and ignore everything else
}

#-----------------------------------------------------------

function Clean-InstalledFolder {
    Write-Output "Cleaning Installed Folder to lower the chance of interfering with the installation process"
    npm cache clean --force

    $foldersToRemove = @(
        "${rootdir}\usr\vendor\ggllm.cpp",
        "${rootdir}\usr\vendor\ggml",
        "${rootdir}\usr\vendor\llama-gguf.cpp",
        "${rootdir}\usr\vendor\llama.cpp",
        "${rootdir}\usr\vendor\whisper.cpp",
        "${rootdir}\usr\node_modules",
        "${env:CONDA_PREFIX}"
    )


    
    foreach ($folder in $foldersToRemove) {
        Remove-Item -Path $folder -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Output "DEBUG HALT ${rootdir}\usr\vendor\ggllm.cpp"
    Start-Sleep -Seconds 0
    Write-Output "Should be done"
}


#--------------------------------------------------------------

function Build-Llama {

    
    # Change directory to llama.cpp
    Set-Location -Path "$rootdir\usr\vendor\llama.cpp" -ErrorAction Stop
    git checkout 93356bd # return to ggml era not the dependencies breaking gguf model mode 

    # Create build directory and change directory to it
    mkdir -Force -Path "build" | Out-Null
    Set-Location -Path "build" -ErrorAction Stop

    # Install dependencies based on the platform
    if ($platform -eq 'Win32NT') {
        #Install-DependenciesWindows
    } elseif ($platform -eq 'Unix') {
        Install-DependenciesLinux
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }

    $cuda = Detect-CUDA
    $opencl = Detect-OpenCL

    if ($platform -eq 'Win32NT') {
        if ($cuda -eq "cuda") {
            $env:CMAKE_ARGS += " -DLLAMA_CUBLAS=on"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($arch -eq "amd64" -and $metal -eq "metal") {
            $env:CMAKE_ARGS += " -DLLAMA_METAL=on"
        } elseif (($arch -eq "arm64" -or $arch -eq "aarch64") -and $metal -eq "metal") {
            $env:CMAKE_ARGS += " -DLLAMA_METAL=on"
        } elseif ($arch -eq "amd64" -and $opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_CLBLAST=on"
        } elseif (($arch -eq "arm64" -or $arch -eq "aarch64") -and $opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_CLBLAST=on"
        } elseif ($arch -eq "amd64" -and $opencl -eq "no_opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        } elseif (($arch -eq "arm64" -or $arch -eq "aarch64") -and $opencl -eq "no_opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        } else {
            Write-Output "No special Acceleration, Ignoring"
        }
    } elseif ($platform -eq 'Unix') {
        if ($cuda -eq "cuda") {
            $env:CMAKE_ARGS += " -DLLAMA_CUBLAS=on"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($metal -eq "metal") {
            $env:CMAKE_ARGS += " -DLLAMA_METAL=on"
        } elseif ($opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_CLBLAST=on"
        } else {
            $env:CMAKE_ARGS += " -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        }

        if ((Get-WmiObject -Class Win32_Processor).Architecture -eq "ARM64") {
            Write-Output "Enforcing compilation to ARM64, Probably cmake wont listen!"
            $env:CMAKE_HOST_SYSTEM_PROCESSOR = "arm64"
            $ENFORCE_ARCH_COMPILATION = "-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        } else {
            $ENFORCE_ARCH_COMPILATION = ""
        }
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }
    
    # Run CMake
    Write-Output "$env:CMAKE_ARGS $env:CMAKE_CUDA_FLAGS"
    cmake .. $env:CMAKE_ARGS $env:CMAKE_CUDA_FLAGS $ENFORCE_ARCH_COMPILATION
    
    # Build with multiple cores
    Write-Output "This is the architecture $(Get-WmiObject -Class Win32_Processor).Architecture unless the cmake becoming asshole and detect ARM64 as x86_64"
    cmake --build . --config Release --parallel 128
    Set-Location -Path $rootdir -ErrorAction Stop
}

#------------------------------------------------------------------

function Build-Llama-Gguf {
    # Change directory to llama.cpp
    Set-Location -Path "$rootdir\usr\vendor\llama-gguf.cpp" -ErrorAction Stop

    # Create build directory and change directory to it
    mkdir -Force -Path "build" | Out-Null
    Set-Location -Path "build" -ErrorAction Stop

    # Install dependencies based on the platform
    if ($platform -eq 'Win32NT') {
        #Install-DependenciesWindows
    } elseif ($platform -eq 'Unix') {
        Install-DependenciesLinux
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }

    $cuda = Detect-CUDA
    $opencl = Detect-OpenCL
    

    if ($platform -eq 'Win32NT') {
        if ($cuda -eq "cuda") {
            $env:CMAKE_ARGS += " -DLLAMA_CUBLAS=on"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($arch -eq "amd64" -and $metal -eq "metal") {
            $env:CMAKE_ARGS += " -DLLAMA_METAL=on"
        } elseif (($arch -eq "arm64" -or $arch -eq "aarch64") -and $metal -eq "metal") {
            $env:CMAKE_ARGS += " -DLLAMA_METAL=on"
        } elseif ($arch -eq "amd64" -and $opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_CLBLAST=on"
        } elseif (($arch -eq "arm64" -or $arch -eq "aarch64") -and $opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_CLBLAST=on"
        } elseif ($arch -eq "amd64" -and $opencl -eq "no_opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        } elseif (($arch -eq "arm64" -or $arch -eq "aarch64") -and $opencl -eq "no_opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        } else {
            Write-Output "No special Acceleration, Ignoring"
        }
    } elseif ($platform -eq 'Unix') {
        if ($cuda -eq "cuda") {
            $env:CMAKE_ARGS += " -DLLAMA_CUBLAS=on"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($metal -eq "metal") {
            $env:CMAKE_ARGS += " -DLLAMA_METAL=on"
        } elseif ($opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DLLAMA_CLBLAST=on"
        } else {
            $env:CMAKE_ARGS += " -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        }

        if ((Get-WmiObject -Class Win32_Processor).Architecture -eq "ARM64") {
            Write-Output "Enforcing compilation to ARM64, Probably cmake wont listen!"
            $env:CMAKE_HOST_SYSTEM_PROCESSOR = "arm64"
            $ENFORCE_ARCH_COMPILATION = "-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        } else {
            $ENFORCE_ARCH_COMPILATION = ""
        }
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }

    # Run CMake
    Write-Output "$env:CMAKE_ARGS $env:CMAKE_CUDA_FLAGS"
    cmake .. $env:CMAKE_ARGS $env:CMAKE_CUDA_FLAGS

    # Build with multiple cores
    cmake --build . --config Release --parallel 128
    Set-Location -Path $rootdir -ErrorAction Stop
}

#---------------------------------------------------------------------------

function Build-Ggml-Base {
    Write-Output "Requesting GGML Binary"

    
    # Change directory to llama.cpp
    Set-Location -Path "$rootdir\usr\vendor\ggml" -ErrorAction Stop

    # Create build directory and change directory to it
    mkdir -Force -Path "build" | Out-Null
    Set-Location -Path "build" -ErrorAction Stop

    # Install dependencies based on the platform
    if ($platform -eq 'Win32NT') {
        #Install-DependenciesWindows
    } elseif ($platform -eq 'Unix') {
        Install-DependenciesLinux
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }

    $cuda = Detect-CUDA
    $opencl = Detect-OpenCL
    

    if ($platform -eq 'Win32NT') {
        if ($cuda -eq "cuda") {
            $env:CMAKE_ARGS += " -DGGML_CUBLAS=on"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($arch -eq "amd64" -and $metal -eq "metal") {
            $env:CMAKE_ARGS += " -DGGML_METAL=on"
        } elseif ($arch -eq "arm64" -and $metal -eq "metal") {
            $env:CMAKE_ARGS += " -DGGML_METAL=on"
        } elseif ($arch -eq "amd64" -and $opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DGGML_CLBLAST=on"
        } elseif ($arch -eq "arm64" -and $opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DGGML_CLBLAST=on"
        } elseif ($arch -eq "amd64" -and $opencl -eq "no_opencl") {
            $env:CMAKE_ARGS += " -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        } elseif ($arch -eq "arm64" -and $opencl -eq "no_opencl") {
            $env:CMAKE_ARGS += " -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        } else {
            Write-Output "No special Acceleration, Ignoring"
        }
    } elseif ($platform -eq 'Unix') {
        if ($cuda -eq "cuda") {
            $env:CMAKE_ARGS += " -DGGML_CUBLAS=on"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($metal -eq "metal") {
            $env:CMAKE_ARGS += " -DGGML_METAL=on"
        } elseif ($opencl -eq "opencl") {
            $env:CMAKE_ARGS += " -DGGML_CLBLAST=on"
        } else {
            $env:CMAKE_ARGS += " -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
        }

        if ((Get-WmiObject -Class Win32_Processor).Architecture -eq "ARM64") {
            Write-Output "Enforcing compilation to ARM64, Probably cmake wont listen!"
            $env:CMAKE_HOST_SYSTEM_PROCESSOR = "arm64"
            $ENFORCE_ARCH_COMPILATION = "-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        } else {
            $ENFORCE_ARCH_COMPILATION = ""
        }
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }

    # Run CMake
    Write-Output "$env:CMAKE_ARGS $env:CMAKE_CUDA_FLAGS"
    cmake .. $env:CMAKE_ARGS $env:CMAKE_CUDA_FLAGS

    # Build with multiple cores
    make -j 128 ${1}
    Set-Location -Path $rootdir -ErrorAction Stop
}

#----------------------------------------------------------------------------

function Build-Gemma-Base {
    Write-Output "Requesting Google Gemma Binary"

    
    # Change directory to llama.cpp
    Set-Location -Path "$rootdir\usr\vendor\gemma.cpp" -ErrorAction Stop

    # Create build directory and change directory to it
    if (-not (Test-Path "build" -PathType Container)) {
        mkdir -Force -Path "build" | Out-Null
    }
    Set-Location -Path "build" -ErrorAction Stop

    # Install dependencies based on the platform
    if ($platform -eq 'Win32NT') {
        #Install-DependenciesWindows
    } elseif ($platform -eq 'Unix') {
        Install-DependenciesLinux
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }

    $cuda = Detect-CUDA
    $opencl = Detect-OpenCL
    

    if ($platform -eq 'Win32NT') {
        if ($cuda -eq "cuda") {
            $CMAKE_ARGS = ""
            Write-Output "Gemma is CPU SIMD only!"
        }
    } elseif ($platform -eq 'Unix') {
        if ($cuda -eq "cuda") {
            $CMAKE_ARGS = ""
            Write-Output "Gemma is CPU SIMD only!"
            # $env:CMAKE_CUDA_FLAGS = "-allow-unsupported-compiler"
        } elseif ($metal -eq "metal") {
            $CMAKE_ARGS = ""
            Write-Output "Gemma is CPU SIMD only!"
        } elseif ($opencl -eq "opencl") {
            $CMAKE_ARGS = ""
            Write-Output "Gemma is CPU SIMD only!"
        } else {
            $CMAKE_ARGS = ""
            Write-Output "Gemma is CPU SIMD only!"
        }
        if ((Get-WmiObject -Class Win32_Processor).Architecture -eq "ARM64") {
            Write-Output "Enforcing compilation to ARM64, Probably cmake wont listen!"
            $env:CMAKE_HOST_SYSTEM_PROCESSOR = "arm64"
            $ENFORCE_ARCH_COMPILATION = "-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DCMAKE_HOST_SYSTEM_PROCESSOR=arm64 -DCMAKE_SYSTEM_PROCESSOR=arm64"
        } else {
            $ENFORCE_ARCH_COMPILATION = ""
        }
    } else {
        Write-Output "Unsupported platform: $platform"
        exit 1
    }
    # Run CMake
    cmake .. $CMAKE_ARGS $CMAKE_CUDA_FLAGS

    # Build with multiple cores
    make -j 128 ${1}
    Set-Location -Path $rootdir -ErrorAction Stop
}

#---------------------------------------------------------------

# Select working mode
if ($args[0] -eq "llama") {
    # Check if LLaMa is already compiled
    if ((Test-Path ".\usr\vendor\llama.cpp\build\bin\main") -or (Test-Path ".\usr\vendor\llama.cpp\build\bin\main.exe")) {
        Write-Output "${args[0]} binary already compiled. Moving it to .\usr\bin\..."
        if (Test-Path ".\usr\vendor\llama.cpp\build\bin\${args[0]}") {
            Copy-Item ".\usr\vendor\llama.cpp\build\bin\main" ".\usr\bin\chat"
        }
        if (Test-Path ".\usr\vendor\ggml\build\bin\${args[0]}.exe") {
            Copy-Item ".\usr\vendor\llama.cpp\build\bin\main.exe" ".\usr\bin\chat.exe"
        }
    }
    else {
        # LLaMa not compiled, build it
        Write-Output "LLaMa binary not found. Building LLaMa..."
        Build-Llama
    }
}

if ($args[0] -eq "mpt") {
    # Check if ggml Binary already compiled
    Write-Output "Requested Universal GGML Binary Mode"
    Write-Output "MPT ggml no longer exists and integrated with the LLaMa-2 engine!"
}

if ($args[0] -eq "dolly-v2" -or $args[0] -eq "gpt-j" -or $args[0] -eq "gpt-neox" -or $args[0] -eq "mnist" -or $args[0] -eq "replit" -or $args[0] -eq "starcoder" -or $args[0] -eq "whisper") {
    # Check if ggml Binary already compiled
    Write-Output "Requested Universal GGML Binary Mode"
    if ((Test-Path ".\usr\vendor\ggml\build\bin\$($args[0])") -or (Test-Path ".\usr\vendor\ggml\build\bin\$($args[0]).exe")) {
        Write-Output "$($args[0]) binary already compiled. Moving it to .\usr\bin\..."
        if (Test-Path ".\usr\vendor\ggml\build\bin\$($args[0])") {
            Copy-Item ".\usr\vendor\ggml\build\bin\$($args[0])" ".\usr\bin\chat"
        }
        if (Test-Path ".\usr\vendor\ggml\build\bin\$($args[0]).exe") {
            Copy-Item ".\usr\vendor\ggml\build\bin\$($args[0]).exe" ".\usr\bin\chat.exe"
        }
    }
    else {
        # LLaMa not compiled, build it
        Write-Output "$($args[0]) binary not found. Building $($args[0])..."
        Build-Ggml-Base "$($args[0])"
    }
}

if ($args[0] -eq "gpt-2") {
    # Check if ggml Binary already compiled
    Write-Output "Requested Universal GGML Binary Mode"
    Write-Output "gpt-2 is no longer available!"
}

if ($args[0] -eq "falcon") {
    # Check if Falcon ggllm.cpp compiled
    if ((Test-Path ".\vendor\llama.cpp\build\bin\main") -or (Test-Path ".\vendor\llama.cpp\build\bin\main.exe")) {
        Write-Output "falcon binary already compiled. Moving it to .\usr\bin\..."
        Copy-Item ".\vendor\ggllm.cpp\build\bin\main" ".\usr\bin\chat"
        Copy-Item ".\vendor\ggllm.cpp\build\bin\main.exe" ".\usr\bin\chat.exe"
    }
    else {
        # LLaMa not compiled, build it
        Write-Output "Falcon Not found! Compiling"
        Build-Falcon
    }
}

#----------------------------------------------------------------------------------
function Build-LLMBackend {
    # Compile all binaries with specific version and support
    
    Write-Output "Platform: $platform, Architecture: $arch"
    
    $targetFolderArch = $arch
    if ($arch -eq "aarch64") {
        $targetFolderArch = "arm64"
    }

    if ($arch -eq "x86_64") {
        $targetFolderArch = "x64"
    }

    $targetFolderPlatform = ""
    if ($platform -eq "Darwin") {
        $targetFolderPlatform = "0_macOS"
    }

    if ($platform -eq "Linux") {
        $targetFolderPlatform = "2_Linux"
    }

    # Since the cuda binaries or the opencl binaries can be used as the nonaccel binaries too, we can just copy the same binaries to the folder
    # This naming system was introduced due to the Windows different LLMBackend precompiled versions (check llama.cpp and ggllm.cpp release tabs and see the different version of version)
    # example directory ./usr/bin/0_macOS/arm64/LLMBackend-llama-noaccel

    # You know what let's abandon Windows enforced binary structuring and find another way on how to execute other way to have specific acceleration on Windows
    Set-Location -Path $rootdir
    Build-LLama
    Set-Location -Path $rootdir
    Write-Output "Cleaning binaries Replacing with new ones"
    Remove-Item -Path "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch" -Recurse -Force -ErrorAction SilentlyContinue
    if (!(Test-Path "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch")) {
        New-Item -Path "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch" -ItemType Directory | Out-Null
        Write-Output "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch"
    }
    
    mkdir "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\llama"
    Copy-Item ".\usr\vendor\llama.cpp\build\bin\main" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\llama\LLMBackend-llama"
    Write-Output "Copying any Acceleration and Debugging Dependencies for LLaMa GGML v2 v3 Legacy"
    Copy-Item -Recurse ".\usr\vendor\llama.cpp\build\bin\*" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\llama"

    Set-Location -Path $rootdir
    Build-LLama-GGUF
    Set-Location -Path $rootdir
    mkdir "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\llama-gguf"
    Copy-Item ".\usr\vendor\llama-gguf.cpp\build\bin\main" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\llama-gguf\LLMBackend-llama-gguf"
    Write-Output "Copying any Acceleration and Debugging Dependencies for LLaMa GGUF Neo Model"
    Copy-Item -Recurse ".\usr\vendor\llama-gguf.cpp\build\bin\*" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\llama-gguf"

    Set-Location -Path $rootdir
    Build-Falcon
    Set-Location -Path $rootdir
    mkdir "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\falcon"
    Copy-Item ".\usr\vendor\ggllm.cpp\build\bin\main" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\falcon\LLMBackend-falcon"
    Write-Output "Copying any Acceleration and Debugging Dependencies for Falcon"
    Copy-Item -Recurse ".\usr\vendor\ggllm.cpp\build\bin\*" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\falcon"

    Set-Location -Path $rootdir
    Build-Ggml-Base "gpt-j"
    Set-Location -Path $rootdir
    # ./usr/vendor/ggml/build/bin/${1} location of the compiled binary ggml based
    mkdir "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\ggml-gptj"
    Write-Output "Copying any Acceleration and Debugging Dependencies for gpt-j"
    Copy-Item -Recurse ".\usr\vendor\ggml\build\bin\*" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\ggml-gptj"
    Copy-Item ".\usr\vendor\ggml\build\bin\gpt-j" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\ggml-gptj\LLMBackend-gpt-j"

    Set-Location -Path $rootdir
    # gemma
    Build-Gemma-Base
    Set-Location -Path $rootdir
    # ./usr/vendor/ggml/build/bin/${1} location of the compiled binary ggml based
    mkdir "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\googleGemma"
    Write-Output "Copying any Acceleration and Debugging Dependencies for googleGemma"
    Copy-Item -Recurse ".\usr\vendor\gemma.cpp\build\*" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\googleGemma"
    Copy-Item ".\usr\vendor\gemma.cpp\build\gemma" "$rootdir\usr\bin\$targetFolderPlatform\$targetFolderArch\googleGemma\LLMBackend-gemma"

    Set-Location -Path $rootdir
}

#--------------------------------------------------------------------------

# Change directory to .\usr and install npm dependencies
Set-Location -Path ".\usr" -ErrorAction Stop

#-------------------------------------------------------------------------

# Install npm dependency if npm is not installed
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    if ($env:PLATFORM -eq "Windows_NT") {
        # Install npm dependencies for Windows 10 22h2 and beyond
        # Add your installation commands here
    }
}
#--------------------------------------------------------------------------

# Install npm dependencies if installed.flag file does not exist or FORCE_REBUILD is set to 1
if ((-not (Test-Path "${rootdir}\installed.flag")) -or (${env:FORCE_REBUILD} -eq "1")) {
    Clean-InstalledFolder
    Install-DependenciesWindows
    Write-Output "Enforcing latest npm"
    npm install npm@latest
    Write-Output "Installing Modules"
    npm install --save-dev
    npm audit fix
    npx --openssl_fips='' electron-rebuild
    Import-SubmoduleManually 
    Build-LLMBackend
    New-Item -ItemType File -Path "${rootdir}\installed.flag"
}

#----------------------------------------------------------------------------

# Change directory to ${rootdir}\usr
Set-Location "${rootdir}\usr"

# Check Node.js version
node -v

# Check npm version
npm -v

# Start npm
npm start
