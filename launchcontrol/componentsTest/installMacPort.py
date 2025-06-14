import os
import platform
import subprocess
import urllib.request

def get_macos_version():
    return platform.mac_ver()[0]

def find_latest_macports_version():
    url = "https://distfiles.macports.org/MacPorts/"
    with urllib.request.urlopen(url) as response:
        html = response.read().decode('utf-8')
    versions = []
    for line in html.splitlines():
        if "MacPorts-" in line and ".pkg" in line:
            start_idx = line.find("MacPorts-") + len("MacPorts-")
            end_idx = line.find(".pkg")
            version = line[start_idx:end_idx]
            versions.append(version)
    return sorted(versions, reverse=True)[0] if versions else None

def download_macports(version, macos_version):
    #url = f"https://distfiles.macports.org/MacPorts/MacPorts-{version}-{macos_version}.pkg"
    url = f"https://distfiles.macports.org/MacPorts/MacPorts-{version}.pkg"
    print(url)
    file_path = f"/tmp/MacPorts-{version}-{macos_version}.pkg"
    urllib.request.urlretrieve(url, file_path)
    return file_path

def install_macports(pkg_path):
    subprocess.run(['sudo', 'installer', '-pkg', pkg_path, '-target', '/'], check=True)

def main():
    macos_version = get_macos_version()
    print(f"Detected macOS version: {macos_version}")

    latest_version = find_latest_macports_version()
    if not latest_version:
        print("Could not find the latest MacPorts version.")
        return

    print(f"Latest MacPorts version: {latest_version}")

    pkg_path = download_macports(latest_version, macos_version)
    print(f"Downloaded MacPorts package to: {pkg_path}")

    install_macports(pkg_path)
    print("MacPorts installed successfully.")

if __name__ == "__main__":
    main()
