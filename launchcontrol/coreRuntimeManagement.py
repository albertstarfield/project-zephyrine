#!/usr/bin/python3
import os
import platform
import subprocess
import sys
import multiprocessing
import shutil
import time
import getpass

# Define global variables
rootdir = os.getcwd()
CONDA_PREFIX = os.path.join(rootdir, 'conda_python_modules')
LC_CTYPE = 'UTF-8'
N_PREFIX = os.path.join(rootdir, 'usr', 'nodeLocalRuntime')

# Update environment variables
os.environ['CONDA_PREFIX'] = CONDA_PREFIX
os.environ['LC_CTYPE'] = LC_CTYPE
os.environ['N_PREFIX'] = N_PREFIX
os.environ['PATH'] = os.pathsep.join([os.path.join(N_PREFIX, 'bin'), os.path.join(CONDA_PREFIX, 'bin'), os.environ['PATH']])
global login   
login = "UNK" 
def superuserPreventionWorkaround_FalseUserWoraround(): 
    #https://forum.sublimetext.com/t/os-getlogin-root-wrong/49442
    #https://github.com/kovidgoyal/kitty/issues/6511
    global login  #forward modified variable
    #login = os.getlogin() #os.getlogin() doesn't work on some glibc https://bugs.python.org/issue40821
    login = getpass.getuser()
    print(login)


superuserPreventionWorkaround_FalseUserWoraround() #setting login variable as the replacement of broken os.getlogin()
print(login)
targetuserFixPerm=login + ":" + "admin"
print(targetuserFixPerm)
time.sleep(1)

def check_superuser():
    if os.name == 'nt':
        import ctypes
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    else:
        return os.geteuid() == 0



def check_relaunch_as_bash():
    if sys.argv[0].endswith('sh'):
        os.execvp('/bin/bash', ['bash'] + sys.argv)

def get_alloc_threads():
    if platform.system() == 'Darwin':
        return os.cpu_count() // 4
    elif platform.system() == 'Linux':
        return os.cpu_count() // 4
    elif platform.system() == 'Windows':
        return os.cpu_count() // 4
    else:
        return 1

def detect_cuda():
    if os.environ.get('ENFORCE_NOACCEL', '0') != "1":
        try:
            subprocess.run(["nvcc", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "cuda"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return "no_cuda"

def detect_opencl():
    if os.environ.get('ENFORCE_NOACCEL', '0') != "1":
        if platform.system() == 'Windows':
            if os.path.exists("C:\\Windows\\System32\\OpenCL.dll"):
                return "opencl"
            else:
                return "no_opencl"
        elif platform.system() == 'Linux':
            if os.path.exists("/usr/include/CL/cl.h"):
                return "opencl"
            else:
                return "no_opencl"
        elif platform.system() == 'Darwin':
            if os.path.exists("/System/Library/Frameworks/OpenCL.framework"):
                return "opencl"
            else:
                return "no_opencl"
        else:
            return "unsupported"
    return "no_opencl"

def detect_metal():
    if os.environ.get('ENFORCE_NOACCEL', '0') == '1':
        return "no_metal"
    if platform.system() == 'Darwin':
        if os.path.exists("/System/Library/Frameworks/Metal.framework"):
            return "metal"
        return "no_metal"
    return "unsupported"


def install_dependencies_linux():
    package_manager = None
    if subprocess.call(["which", "apt-get"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        package_manager = 'apt-get'
    elif subprocess.call(["which", "dnf"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        package_manager = 'dnf'
    elif subprocess.call(["which", "yum"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        package_manager = 'yum'
    elif subprocess.call(["which", "zypper"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        package_manager = 'zypper'
    elif subprocess.call(["which", "pacman"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        package_manager = 'pacman'
    elif subprocess.call(["which", "swupd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        package_manager = 'swupd'

    if package_manager:
        install_commands = {
            'apt-get': ['sudo', 'apt-get', 'update', '&&', 'sudo', 'apt-get', 'install', '-y', 'build-essential', 'python3', 'cmake', 'libopenblas-dev', 'liblapack-dev'],
            'dnf': ['sudo', 'dnf', 'install', '-y', 'gcc-c++', 'cmake', 'openblas-devel', 'python', 'lapack-devel'],
            'yum': ['sudo', 'yum', 'install', '-y', 'gcc-c++', 'cmake', 'openblas-devel', 'python', 'lapack-devel'],
            'zypper': ['sudo', 'zypper', 'install', '-y', 'gcc-c++', 'cmake', 'openblas-devel', 'python', 'lapack-devel'],
            'pacman': ['sudo', 'pacman', '-Syu', '--needed', 'base-devel', 'python', 'cmake', 'openblas', 'lapack'],
            'swupd': ['sudo', 'swupd', 'bundle-add', 'c-basic'],
        }
        subprocess.call(install_commands[package_manager])
    subprocess.call(['npm', 'install', '-g', 'npm@latest'])

def install_dependencies_macos():
    if subprocess.call(["which", "xcode-select"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("Xcode command line tools not found. Please install Xcode and try again.")
        sys.exit(1)
    if subprocess.call(["which", "brew"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("Homebrew not found. Please install Homebrew and try again.")
        sys.exit(1)
    
    subprocess.call(['xcode-select', '--install'])
    subprocess.call(['sudo', 'xcodebuild', '-license', 'accept'])
    print(login)
    targetuserFixPerm=login + ":" + "admin"
    subprocess.call(['sudo', 'chown', '-R', targetuserFixPerm, '/opt/homebrew'])
    subprocess.call(['sudo', 'chown', '-R', targetuserFixPerm, '/usr/local/homebrew'])
    subprocess.call(['brew', 'doctor'])
    subprocess.call(['brew', 'tap', 'homebrew/core'])
    subprocess.call(['brew', 'tap', 'apple/apple'])
    subprocess.call(['brew', 'upgrade', '--greed'])
    subprocess.call(['brew', 'reinstall', 'python', 'node', 'cmake', 'mas'])
    subprocess.call(['mas', 'install', '497799835'])
    subprocess.call(['npm', 'install', '-g', 'n'])
    subprocess.call(['npm', 'install', '-g', 'npm@latest'])
    subprocess.call(['n', 'latest'])

def install_dependencies_windows():
    # Check if Chocolatey is installed, if not, install it
    try:
        subprocess.run(["choco", "-v"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Installing Chocolatey...")
        subprocess.run(["powershell", "Set-ExecutionPolicy", "Bypass", "-Scope", "Process", "-Force"], check=True)
        subprocess.run(["powershell", "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"], check=True)

    # Install cmake with Chocolatey
    subprocess.run(["choco", "install", "cmake", "--installargs", "'ADD_CMAKE_TO_PATH=System'", "-y"], check=True)

    # Install required packages using Chocolatey
    subprocess.run(["choco", "install", "-y", "python3", "git", "nodejs"], check=True)
    subprocess.run(["choco", "install", "-y", "make", "msys2"], check=True)

    # Check if nmake is installed, if not, install it
    try:
        subprocess.run(["nmake", "/?"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Installing build-essential bundle package but for Windows 10+...")
        print("Yes, it's possible to do it without seen or manual GUI on Windows")
        print("Console may stay here for a while until 4-6 GB downloads some necessary dependencies!")
        subprocess.run(["powershell", "Invoke-WebRequest -Uri 'https://aka.ms/vs/16/release/vs_buildtools.exe' -OutFile vs_buildtools.exe"], check=True)
        subprocess.run(["powershell", "Start-Process -FilePath .\\vs_buildtools.exe -ArgumentList '-q --wait --norestart --nocache --installPath C:\\BuildTools --add Microsoft.VisualStudio.Component.VC.CoreIde --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.Component.MSBuild --add Microsoft.VisualStudio.Component.NuGet --add Microsoft.Component.MSBuild --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core --add Microsoft.VisualStudio.Component.VC.CoreIde --add Microsoft.VisualStudio.Component.Roslyn.Compiler' -Wait"], check=True)
        subprocess.run(["powershell", "Start-Sleep -Seconds 10"], check=True)
        
        # Define the path to nmake.exe
        nmake_paths = subprocess.run(["powershell", "(Get-ChildItem -Path 'C:\\BuildTools\\VC\\Tools\\MSVC\\' -Filter '16.*.*' -Directory | Select-Object -ExpandProperty FullName | ForEach-Object { Join-Path -Path $_ -ChildPath 'bin\\Hostx64\\x64' })"], capture_output=True, text=True, check=True).stdout.strip().split()
        
        # Add nmake path to the PATH environment variable
        for path in nmake_paths:
            os.environ["PATH"] += os.pathsep + path
        
        # Check if nmake is now accessible in the PATH
        subprocess.run(["nmake", "/?"], check=True)
    
    # Refresh environment variables
    subprocess.run(["powershell", "Import-Module $env:ChocolateyInstall\\helpers\\chocolateyProfile.psm1; refreshenv"], check=True)

    # Upgrade to the latest Node version
    print("Upgrading to the latest Node Version!")
    subprocess.run(["powershell", "nvm install 20.11.1"], check=True)
    subprocess.run(["powershell", "nvm use 20.11.1"], check=True)
    subprocess.run(["node", "-v"], check=True)



print(os.environ['PATH'])

def clean_installed_folder():
    print("Cleaning Installed Folder to lower the chance of interfering with the installation process")
    
    # Run npm cache clean
    #subprocess.run(['npm', 'cache', 'clean', '--force'], check=True)
    
    # Remove directories
    directories_to_remove = [
        os.path.join(rootdir, 'usr', 'vendor'),
        os.path.join(rootdir, 'usr', 'node_modules'),
        CONDA_PREFIX,
        N_PREFIX
    ]
    
    for directory in directories_to_remove:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    
    # Create the vendor directory
    os.makedirs(os.path.join(rootdir, 'usr', 'vendor'))
    
    print("Should be done")

def clone_submodule(path, url, commit):
    print(f"Cloning submodule: {path} from {url}")
    subprocess.run(['git', 'clone', '--recurse-submodules', '--single-branch', '--branch', commit, url, path], check=True)
    # If you want to use the simple branch checkout, uncomment the following line
    # subprocess.run(['git', 'clone', '--branch', commit, url, path], check=True)

def import_submodules_manually():
    clone_submodule(os.path.join(rootdir, 'usr', 'vendor', 'llama.cpp'), 'https://github.com/ggerganov/llama.cpp', 'master')
    clone_submodule(os.path.join(rootdir, 'usr', 'vendor', 'ggllm.cpp'), 'https://github.com/cmp-nct/ggllm.cpp', 'master')
    clone_submodule(os.path.join(rootdir, 'usr', 'vendor', 'ggml'), 'https://github.com/ggerganov/ggml', 'master')
    clone_submodule(os.path.join(rootdir, 'usr', 'vendor', 'llama-gguf.cpp'), 'https://github.com/ggerganov/llama.cpp', 'master')
    clone_submodule(os.path.join(rootdir, 'usr', 'vendor', 'whisper.cpp'), 'https://github.com/ggerganov/whisper.cpp', 'master')
    clone_submodule(os.path.join(rootdir, 'usr', 'vendor', 'gemma.cpp'), 'https://github.com/google/gemma.cpp', 'main')

#----- Build 

def build_llama():
    os.chdir(os.path.join(rootdir, 'usr', 'vendor', 'llama.cpp'))
    subprocess.run(['git', 'checkout', '93356bd'], check=True)  # return to ggml era not the dependencies breaking gguf model mode

    os.makedirs('build', exist_ok=True)
    os.chdir('build')

    cuda = detect_cuda()
    opencl = detect_opencl()
    metal = detect_metal()

    cmake_args = []
    platform_system = platform.system()
    arch = platform.machine()

    if platform_system == 'Linux':
        if cuda == "cuda":
            cmake_args.append('-DLLAMA_CUBLAS=on')
        elif arch == "x86_64" and metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif arch in ["arm64", "aarch64"] and metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif arch == "x86_64" and opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        elif arch in ["arm64", "aarch64"] and opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        else:
            cmake_args.extend(['-DLLAMA_BLAS=ON', '-DLLAMA_BLAS_VENDOR=OpenBLAS'])
    elif platform_system == 'Darwin':
        if cuda == "cuda":
            cmake_args.append('-DLLAMA_CUBLAS=on')
        elif metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        else:
            cmake_args.extend(['-DLLAMA_BLAS=ON', '-DLLAMA_BLAS_VENDOR=OpenBLAS'])

        if arch == "arm64":
            os.environ["CMAKE_HOST_SYSTEM_PROCESSOR"] = "arm64"
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
            cmake_args.append("-DCMAKE_APPLE_SILICON_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_HOST_SYSTEM_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_SYSTEM_PROCESSOR=arm64")
    else:
        raise RuntimeError(f"Unsupported platform: {platform_system}")

    print(f"CMake arguments: {' '.join(cmake_args)}")
    subprocess.run(['cmake', '..'] + cmake_args, check=True)
    subprocess.run(['cmake', '--build', '.', '--config', 'Release', '--parallel', str(get_alloc_threads())], check=True)

    if platform_system in ['Linux', 'Darwin']:
        subprocess.run(['cp', 'bin/main', os.path.join(rootdir, 'usr', 'bin', 'chat')], check=True)

    os.chdir(rootdir)

def build_llama_gguf():
    os.chdir(os.path.join(rootdir, 'usr', 'vendor', 'llama-gguf.cpp'))

    os.makedirs('build', exist_ok=True)
    os.chdir('build')

    cuda = detect_cuda()
    opencl = detect_opencl()
    metal = detect_metal()

    cmake_args = []
    platform_system = platform.system()
    arch = platform.machine()

    if platform_system == 'Linux':
        if cuda == "cuda":
            cmake_args.append('-DLLAMA_CUBLAS=on')
        elif arch == "x86_64" and metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif arch in ["arm64", "aarch64"] and metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif arch == "x86_64" and opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        elif arch in ["arm64", "aarch64"] and opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        else:
            cmake_args.extend(['-DLLAMA_BLAS=ON', '-DLLAMA_BLAS_VENDOR=OpenBLAS'])
    elif platform_system == 'Darwin':
        if cuda == "cuda":
            cmake_args.append('-DLLAMA_CUBLAS=on')
        elif metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        else:
            cmake_args.extend(['-DLLAMA_BLAS=ON', '-DLLAMA_BLAS_VENDOR=OpenBLAS'])

        if arch == "arm64":
            os.environ["CMAKE_HOST_SYSTEM_PROCESSOR"] = "arm64"
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
            cmake_args.append("-DCMAKE_APPLE_SILICON_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_HOST_SYSTEM_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_SYSTEM_PROCESSOR=arm64")
    else:
        raise RuntimeError(f"Unsupported platform: {platform_system}")

    print(f"CMake arguments: {' '.join(cmake_args)}")
    subprocess.run(['cmake', '..'] + cmake_args, check=True)
    subprocess.run(['cmake', '--build', '.', '--config', 'Release', '--parallel', str(get_alloc_threads())], check=True)

    os.chdir(rootdir)

def build_gemma_base():
    print("Requesting Google Gemma Binary")

    os.chdir(os.path.join(rootdir, 'usr', 'vendor', 'gemma.cpp'))

    if not os.path.exists('build'):
        os.makedirs('build')
    os.chdir('build')

    cuda = detect_cuda()
    opencl = detect_opencl()
    metal = detect_metal()

    cmake_args = []
    platform_system = platform.system()
    arch = platform.machine()

    if platform_system == 'Linux':
        if cuda == "cuda":
            print("Gemma is CPU SIMD only!")
        # No additional CMAKE_ARGS needed for Linux in this context
    elif platform_system == 'Darwin':
        if cuda == "cuda":
            print("Gemma is CPU SIMD only!")
        elif metal == "metal":
            print("Gemma is CPU SIMD only!")
        elif opencl == "opencl":
            print("Gemma is CPU SIMD only!")
        else:
            print("Gemma is CPU SIMD only!")

        if arch == "arm64":
            os.environ["CMAKE_HOST_SYSTEM_PROCESSOR"] = "arm64"
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
            cmake_args.append("-DCMAKE_APPLE_SILICON_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_HOST_SYSTEM_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_SYSTEM_PROCESSOR=arm64")
    else:
        raise RuntimeError(f"Unsupported platform: {platform_system}")

    print(f"CMake arguments: {' '.join(cmake_args)}")
    subprocess.run(['cmake', '..'] + cmake_args, check=True)
    subprocess.run(['cmake', '--build', '.', '--config', 'Release', '--parallel', str(get_alloc_threads())], check=True)

    os.chdir(rootdir)

def build_ggml_base(binary_name):
    print("Requesting GGML Binary")
    
    rootdir = os.getcwd()
    platform_system = platform.system()
    arch = platform.machine()
    
    try:
        os.chdir('usr/vendor/ggml')
    except FileNotFoundError:
        print("Directory not found: usr/vendor/ggml")
        return
    
    os.makedirs('build', exist_ok=True)
    os.chdir('build')

    cuda = detect_cuda()
    opencl = detect_opencl()
    metal = "no_metal"  # Placeholder, update this if you have a metal detection function

    cmake_args = []

    if platform_system == 'Linux':
        if cuda == "cuda":
            cmake_args.append("-DGGML_CUBLAS=on")
        elif arch == "x86_64" and metal == "metal":
            cmake_args.append("-DGGML_METAL=on")
        elif arch in ["arm64", "aarch64"] and metal == "metal":
            cmake_args.append("-DGGML_METAL=on")
        elif arch == "x86_64" and opencl == "opencl":
            cmake_args.append("-DGGML_CLBLAST=on")
        elif arch in ["arm64", "aarch64"] and opencl == "opencl":
            cmake_args.append("-DGGML_CLBLAST=on")
        elif arch == "x86_64" and opencl == "no_opencl":
            cmake_args.append("-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS")
        elif arch in ["arm64", "aarch64"] and opencl == "no_opencl":
            cmake_args.append("-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS")
        else:
            print("No special Acceleration, Ignoring")
    elif platform_system == 'Darwin':
        if cuda == "cuda":
            cmake_args.append("-DGGML_CUBLAS=on")
        elif metal == "metal":
            cmake_args.append("-DGGML_METAL=on")
        elif opencl == "opencl":
            cmake_args.append("-DGGML_CLBLAST=on")
        else:
            cmake_args.append("-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS")
        if arch == "arm64":
            os.environ["CMAKE_HOST_SYSTEM_PROCESSOR"] = "arm64"
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
            cmake_args.append("-DCMAKE_APPLE_SILICON_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_SYSTEM_PROCESSOR=arm64")
    else:
        print("Unsupported platform:", platform_system)
        return

    try:
        subprocess.run(["cmake", ".."] + cmake_args, check=True)
        subprocess.run(["make", "-j", str(get_alloc_threads()), binary_name], check=True)
        os.chdir(rootdir)
        if platform_system == 'Linux':
            subprocess.run(["cp", f"bin/{binary_name}", f"{rootdir}/usr/bin/chat"], check=True)
        elif platform_system == 'Darwin':
            subprocess.run(["cp", f"bin/{binary_name}", f"{rootdir}/usr/bin/chat"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"GGML based compilation failed. See logs for details: {e}")
    finally:
        os.chdir(rootdir)

def build_falcon():
    os.chdir(os.path.join(rootdir, 'usr', 'vendor', 'ggllm.cpp'))

    os.makedirs('build', exist_ok=True)
    os.chdir('build')

    cuda = detect_cuda()
    opencl = detect_opencl()
    metal = detect_metal()

    cmake_args = []
    platform_system = platform.system()
    arch = platform.machine()

    if platform_system == 'Linux':
        if cuda == "cuda":
            cmake_args.append('-DLLAMA_CUBLAS=on')
        elif arch == "x86_64" and metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif arch in ["arm64", "aarch64"] and metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif arch == "x86_64" and opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        elif arch in ["arm64", "aarch64"] and opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        else:
            cmake_args.extend(['-DLLAMA_BLAS=ON', '-DLLAMA_BLAS_VENDOR=OpenBLAS'])
    elif platform_system == 'Darwin':
        if cuda == "cuda":
            cmake_args.append('-DLLAMA_CUBLAS=on')
        elif metal == "metal":
            cmake_args.append('-DLLAMA_METAL=on')
        elif opencl == "opencl":
            cmake_args.append('-DLLAMA_CLBLAST=on')
        else:
            cmake_args.extend(['-DLLAMA_BLAS=ON', '-DLLAMA_BLAS_VENDOR=OpenBLAS'])

        if arch == "arm64":
            os.environ["CMAKE_HOST_SYSTEM_PROCESSOR"] = "arm64"
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
            cmake_args.append("-DCMAKE_APPLE_SILICON_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_HOST_SYSTEM_PROCESSOR=arm64")
            cmake_args.append("-DCMAKE_SYSTEM_PROCESSOR=arm64")
    else:
        raise RuntimeError(f"Unsupported platform: {platform_system}")

    print(f"CMake arguments: {' '.join(cmake_args)}")
    subprocess.run(['cmake', '..'] + cmake_args, check=True)
    subprocess.run(['cmake', '--build', '.', '--config', 'Release', '--parallel', str(get_alloc_threads())], check=True)

    os.chdir(rootdir)

def buildLLMBackend():
    platform_system = platform.system()
    arch = platform.machine()
    
    print(f"Platform: {platform_system}, Architecture: {arch}")
    print(f"Current Directory: {os.getcwd()}")
    
    targetFolderArch = arch
    if arch == "aarch64":
        targetFolderArch = "arm64"
    elif arch == "x86_64":
        targetFolderArch = "x64"

    if platform_system == "Darwin":
        targetFolderPlatform = "0_macOS"
    elif platform_system == "Linux":
        targetFolderPlatform = "2_Linux"
    else:
        targetFolderPlatform = "unsupported_platform"

    if targetFolderPlatform == "unsupported_platform":
        print(f"Unsupported platform: {platform_system}")
        return

    print("Cleaning binaries and replacing with new ones")
    bin_dir = os.path.join(rootdir, 'usr', 'bin', targetFolderPlatform, targetFolderArch)
    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir)
    os.makedirs(bin_dir, exist_ok=True)

    build_llama()
    
    llama_dir = os.path.join(bin_dir, 'llama')
    os.makedirs(llama_dir, exist_ok=True)
    shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'llama.cpp', 'build', 'bin', 'main'), os.path.join(llama_dir, 'LLMBackend-llama'))
    print("Copying any Acceleration and Debugging Dependencies for LLaMa GGML v2 v3 Legacy")
    shutil.copytree(os.path.join(rootdir, 'usr', 'vendor', 'llama.cpp', 'build', 'bin'), llama_dir, dirs_exist_ok=True)

    build_llama_gguf()
    
    llama_gguf_dir = os.path.join(bin_dir, 'llama-gguf')
    os.makedirs(llama_gguf_dir, exist_ok=True)
    #There are massive changes into the llama-cpp which causes it to not work anymore thus renaming it should be make it less chaotic 
    shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'llama-gguf.cpp', 'build', 'bin', 'llama-cli'), os.path.join(llama_gguf_dir, 'LLMBackend-llama-gguf'))
    shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'llama-gguf.cpp', 'build', 'bin', 'llama-finetune'), os.path.join(llama_gguf_dir, 'finetune'))
    shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'llama-gguf.cpp', 'build', 'bin', 'llama-export-lora'), os.path.join(llama_gguf_dir, 'export-lora'))

    print("Copying any Acceleration and Debugging Dependencies for LLaMa GGUF Neo Model")
    shutil.copytree(os.path.join(rootdir, 'usr', 'vendor', 'llama-gguf.cpp', 'build', 'bin'), llama_gguf_dir, dirs_exist_ok=True)

    build_falcon()
    
    falcon_dir = os.path.join(bin_dir, 'falcon')
    os.makedirs(falcon_dir, exist_ok=True)
    shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'ggllm.cpp', 'build', 'bin', 'main'), os.path.join(falcon_dir, 'LLMBackend-falcon'))
    print("Copying any Acceleration and Debugging Dependencies for Falcon")
    shutil.copytree(os.path.join(rootdir, 'usr', 'vendor', 'ggllm.cpp', 'build', 'bin'), falcon_dir, dirs_exist_ok=True)

    # It's broken so we'll exclude it
    #build_ggml_base('gpt-j')
    
    #ggml_gptj_dir = os.path.join(bin_dir, 'ggml-gptj')
    #os.makedirs(ggml_gptj_dir, exist_ok=True)
    #print("Copying any Acceleration and Debugging Dependencies for gpt-j")
    #shutil.copytree(os.path.join(rootdir, 'usr', 'vendor', 'ggml', 'build', 'bin'), ggml_gptj_dir, dirs_exist_ok=True)
    #shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'ggml', 'build', 'bin', 'gpt-j'), os.path.join(ggml_gptj_dir, 'LLMBackend-gpt-j'))

    build_gemma_base()
    
    google_gemma_dir = os.path.join(bin_dir, 'googleGemma')
    os.makedirs(google_gemma_dir, exist_ok=True)
    print("Copying any Acceleration and Debugging Dependencies for googleGemma")
    shutil.copytree(os.path.join(rootdir, 'usr', 'vendor', 'gemma.cpp', 'build'), google_gemma_dir, dirs_exist_ok=True)
    shutil.copy(os.path.join(rootdir, 'usr', 'vendor', 'gemma.cpp', 'build', 'gemma'), os.path.join(google_gemma_dir, 'LLMBackend-gemma'))


def fix_permission_universal():
    try:
        if platform.system() in ['Linux', 'Darwin']:
            print("Fixing Universal issue on Unix-like system")
            conda_prefix = os.getenv('CONDA_PREFIX', '')
            n_prefix = os.getenv('N_PREFIX', '')
            if conda_prefix and n_prefix:
                subprocess.run(['sudo', 'chmod', '-R', '777', conda_prefix, n_prefix], check=True)
            else:
                print("CONDA_PREFIX or N_PREFIX not set.")
        elif platform.system() == 'Windows':
            print("Fixing Universal issue on Windows")
            conda_prefix = os.getenv('CONDA_PREFIX', '')
            n_prefix = os.getenv('N_PREFIX', '')
            if conda_prefix and n_prefix:
                subprocess.run(['icacls', conda_prefix, '/grant', 'Everyone:F', '/T'], check=True)
                subprocess.run(['icacls', n_prefix, '/grant', 'Everyone:F', '/T'], check=True)
            else:
                print("CONDA_PREFIX or N_PREFIX not set.")
        else:
            print("Unsupported platform.")
    except subprocess.CalledProcessError as e:
        print(f"Error fixing permissions: {e}")

def enforcing_dependencies():
    current_platform = platform.system()
    if current_platform == "Linux":
        install_dependencies_linux()
    elif current_platform == "Darwin":
        install_dependencies_macos()
    elif current_platform == "Windows":
        install_dependencies_windows()
    else:
        print(f"Unsupported platform: {current_platform}")

def main():
    rootdir = os.getcwd()
    force_rebuild = os.getenv("FORCE_REBUILD", "0")
    
    if not os.path.isfile(os.path.join(rootdir, "installed.flag")) or force_rebuild == "1":
        clean_installed_folder()
        print("Enforcing Check of Dependencies!")
        enforcing_dependencies()
        print("Enforcing latest npm")
        os.chdir(os.path.join(rootdir, "usr"))
        subprocess.run(["npm", "install", "npm@latest"], check=True)
        
        print("Installing Modules")
        subprocess.run(["npm", "install", "--save-dev"], check=True)
        
        print("Running npm audit fix")
        subprocess.run(["npm", "audit", "fix"], check=True)
        
        print("Rebuilding Electron Program For specific Computer COnfiguration with OpenSSL FIPS")
        
        print(os.getcwd())
        subprocess.run(["env"], check=True)
        subprocess.run(["npx", "--openssl_fips=''", "electron-rebuild"], check=True)
        #subprocess.run(["env", "&&", "npx", "--openssl_fips=''", "electron-rebuild"], check=True)
        print("Installing Modules")
        #install_modules()
        import_submodules_manually()
        buildLLMBackend()
        fix_permission_universal()
        with open(os.path.join(rootdir, "installed.flag"), 'w') as f:
            f.write("")
    os.chdir(os.path.join(rootdir, "usr"))
    subprocess.run(["node", "-v"], check=True)
    subprocess.run(["npm", "-v"], check=True)
    subprocess.run(["npm", "start"], check=True)

if __name__ == "__main__":
    main()
