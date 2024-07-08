# Define the required directories as an array
$requiredDirs = @("systemCore", "launchcontrol", "launcher-builder-control", "documentation")

# Check if all required directories exist
foreach ($dir in $requiredDirs) {
    if (!(Test-Path -Path $dir -PathType Directory)) {
        Write-Error "Error: $dir directory is missing"
        exit 1
    }
}

# Record the current working directory
$cwd = Get-Location
Write-Host "Current working directory: $cwd"

# Check if Python3 is installed (on Windows, you can use `where` cmdlet)
if (!(Get-Command -Name python3 -ErrorAction SilentlyContinue)) {
    Write-Error "Error: Python3 is not installed"
    exit 1
}

# Create a virtual environment (venv) in the current working directory
& py -m venv _venv

# Activate the virtual environment
. \_venv\Scripts\activate

# Run the launch control script using the virtual environment
& python3 .\launchcontrol\coreRuntimeManagement.py
