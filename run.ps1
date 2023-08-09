Write-Host "Currently Native Windows Support is under development while waiting use the .sh instead (use wsl2)"
exit(1)
# ----------------------------------------------------------------

$ErrorActionPreference = "Stop"
# Check Windows build version
$minWindowsBuild = 16299
$windowsBuild = [System.Environment]::OSVersion.Version.Build
if ($windowsBuild -lt $minWindowsBuild) {
    Write-Host "Windows build version is too low."
    exit
}

# Set working directory and store current working directory
$rootdir = (Get-Location).Path -replace ' ', '` '

# Check platform and architecture
$platform = $env:OSTYPE
$architecture = $env:PROCESSOR_ARCHITECTURE
if ($platform -notin @('darwin', 'linux', 'msys')) {
    Write-Host "Unsupported platform: $platform"
    exit
}

# Check working condition variable
$workingCondition = $args[0]
if ($workingCondition -ne "LLaMa") {
    Write-Host "Invalid working condition."
    exit
}

# Check if Chocolatey and Winget are installed
if (-not (Get-Command "choco" -ErrorAction SilentlyContinue)) {
    Invoke-Expression "& { Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')) }"
}
if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
    Write-Host "Winget not installed. Install it manually."
    exit
}

# Install required packages
$packages = "git", "nodejs", "npm"
if ($platform -eq "msys") {
    $packages += "mingw-w64-x86_64-toolchain"
} elseif ($platform -eq "darwin") {
    $packages += "cmake"
}
foreach ($package in $packages) {
    if ($platform -eq "msys") {
        choco install $package
    } else {
        winget install $package
    }
}

# Additional steps for Windows
if ($platform -eq "msys") {
    if ($architecture -eq "x86") {
        choco install -y visualstudio2019buildtools --package-parameters "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
    } else {
        choco install -y visualstudio2019buildtools
    }
}

# Additional steps for node.js
npm install -g n
n latest
node -v

# Check and move binary for Windows
$binaryPath = Join-Path $rootdir "vendor\llama.cpp\build\bin\main.exe"
$targetBinaryPath = Join-Path $rootdir "usr\bin\chat.exe"
if (Test-Path $binaryPath) {
    Move-Item -Path $binaryPath -Destination $targetBinaryPath -Force
} else {
    # Function to compile binary
    function CompileBinary {
        # Initialize variables and check platform and architecture
        $isWindows = ($platform -eq "msys")
        $isWine = ($env:WINEPREFIX -ne $null)
    
        if ($isWindows -and -not $isWine) {
            # Windows-specific compilation steps
            Write-Host "Compiling on Windows..."
    
            # Step 1: Git submodule update
            git submodule update --init --recursive
    
            # Step 2: Change directory to llama.cpp
            Set-Location (Join-Path $rootdir "usr\vendor\llama.cpp")
    
            # Step 3: Create build directory
            New-Item -Path "build" -ItemType "directory"
    
            # Step 4: Change directory to build
            Set-Location (Join-Path $rootdir "usr\vendor\llama.cpp\build")
    
            # Step 5: Install dependencies
            if ($isWindows -and $architecture -eq "x86_64") {
                # Install 64-bit dependencies
                choco install -y build-essential cmake openblas cublas
            } elseif ($isWindows -and $architecture -eq "x86") {
                # Install 32-bit dependencies
                choco install -y mingw-w64-x86_64-toolchain cmake openblas clblast
            }
            
            # Step 6: Run CMake
            $cmakeArgs = ""
            if ($architecture -eq "x86_64") {
                $cmakeArgs = "-DLLAMA_CUBLAS=on"
            } elseif ($architecture -eq "x86") {
                $cmakeArgs = "-DLLAMA_CLBLAST=on"
            }
            cmake .. $cmakeArgs
    
            # Step 7: Build using CMake
            cmake --build . --config Release --parallel $(nproc)
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Compilation failed. Check logs for details."
                exit 1
            }
    
            # Step 8: Change directory back to rootdir
            Set-Location $rootdir
    
            # Step 9: Move binary
            Move-Item -Path "usr\vendor\llama.cpp\build\bin\main.exe" -Destination "usr\bin\chat.exe" -Force
        } else {
            Write-Host "Unsupported compilation environment."
        }
    }
    
    CompileBinary
}

# Install npm dependencies
Set-Location (Join-Path $rootdir "usr")
npm install --save-dev
npm run win-x64 -ErrorAction Stop

# Launch the application
$zephyrinePath = Join-Path $rootdir "usr\release-builds\*Zephyrine*\Project Zephyrine.exe"
Set-Location $zephyrinePath
Start-Process -FilePath "Project Zephyrine.exe" -ArgumentList "--no-sandbox --disable-dev-shm-usage"
