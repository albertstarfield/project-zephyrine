#!/usr/bin/env python3
"""
Package Manager Tool - Install system packages for Adelaide Lite.

Usage: python3 package.py <command> [args...]

Commands:
  install <package>     - Install a package
  uninstall <package>   - Uninstall a package
  update                - Update package lists
  upgrade               - Upgrade all packages
  search <query>        - Search for packages
  list                  - List installed packages
  detect                - Detect available package manager

Supported Package Managers:
  Linux: apt/dpkg, yum/rpm, pacman, emerge, zypper, apk, xbps, nix
  macOS: brew, port
  Windows: winget, choco, scoop

DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import subprocess
import sys
import os
import platform


def detect_package_manager():
    """Detect the available package manager."""
    system = platform.system().lower()
    
    if system == "linux":
        # Check for various Linux package managers
        managers = [
            ("apt", ["apt-get", "install", "-y"]),
            ("yum", ["yum", "install", "-y"]),
            ("dnf", ["dnf", "install", "-y"]),
            ("pacman", ["pacman", "-S", "--noconfirm"]),
            ("emerge", ["emerge"]),
            ("zypper", ["zypper", "install", "-y"]),
            ("apk", ["apk", "add"]),
            ("xbps", ["xbps-install", "-y"]),
            ("nix", ["nix-env", "-iA", "nixpkgs"]),
        ]
        
        for name, cmd in managers:
            if cmd_exists(cmd[0]):
                return name, cmd
    
    elif system == "darwin":
        # macOS
        if cmd_exists("brew"):
            return "brew", ["brew", "install"]
        elif cmd_exists("port"):
            return "port", ["port", "install"]
    
    elif system == "windows":
        # Windows
        if cmd_exists("winget"):
            return "winget", ["winget", "install", "--accept-package-agreements"]
        elif cmd_exists("choco"):
            return "choco", ["choco", "install", "-y"]
        elif cmd_exists("scoop"):
            return "scoop", ["scoop", "install"]
    
    return None, None


def cmd_exists(cmd):
    """Check if a command exists."""
    try:
        result = subprocess.run(
            ["which", cmd] if platform.system() != "Windows" else ["where", cmd],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def run_cmd(cmd, sudo=False):
    """Run a command."""
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out"
    except Exception as e:
        return f"ERROR: {e}"


def install_package(package):
    """Install a package using the detected package manager."""
    name, base_cmd = detect_package_manager()
    
    if not name:
        return "ERROR: No supported package manager found"
    
    print(f"Detected package manager: {name}")
    print(f"Installing: {package}")
    
    # Special handling for some package managers
    if name == "apt":
        # Update first
        run_cmd(["apt-get", "update"], sudo=True)
        cmd = base_cmd + [package]
    elif name == "pacman":
        # Sync first
        run_cmd(["pacman", "-Sy"], sudo=True)
        cmd = base_cmd + [package]
    else:
        cmd = base_cmd + [package]
    
    output = run_cmd(cmd, sudo=True)
    return output


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "detect":
        name, base_cmd = detect_package_manager()
        if name:
            print(f"Package manager: {name}")
            print(f"Install command: {' '.join(base_cmd)}")
        else:
            print("No supported package manager found")

    elif cmd == "install":
        if not args:
            print("ERROR: Usage: package.py install <package>")
            return 1
        output = install_package(args[0])
        print(output)

    elif cmd == "uninstall":
        if not args:
            print("ERROR: Usage: package.py uninstall <package>")
            return 1
        name, base_cmd = detect_package_manager()
        if not name:
            print("ERROR: No supported package manager found")
            return 1
        
        # Modify command for uninstall
        if name == "apt":
            cmd = ["apt-get", "remove", "-y", args[0]]
        elif name == "yum" or name == "dnf":
            cmd = [name, "remove", "-y", args[0]]
        elif name == "pacman":
            cmd = ["pacman", "-R", "--noconfirm", args[0]]
        elif name == "brew":
            cmd = ["brew", "uninstall", args[0]]
        else:
            print(f"ERROR: Uninstall not implemented for {name}")
            return 1
        
        output = run_cmd(cmd, sudo=True)
        print(output)

    elif cmd == "update":
        name, _ = detect_package_manager()
        if not name:
            print("ERROR: No supported package manager found")
            return 1
        
        if name == "apt":
            output = run_cmd(["apt-get", "update"], sudo=True)
        elif name == "yum":
            output = run_cmd(["yum", "check-update"], sudo=True)
        elif name == "dnf":
            output = run_cmd(["dnf", "check-update"], sudo=True)
        elif name == "pacman":
            output = run_cmd(["pacman", "-Sy"], sudo=True)
        elif name == "brew":
            output = run_cmd(["brew", "update"])
        else:
            output = f"Update not implemented for {name}"
        
        print(output)

    elif cmd == "upgrade":
        name, _ = detect_package_manager()
        if not name:
            print("ERROR: No supported package manager found")
            return 1
        
        if name == "apt":
            output = run_cmd(["apt-get", "upgrade", "-y"], sudo=True)
        elif name == "yum":
            output = run_cmd(["yum", "update", "-y"], sudo=True)
        elif name == "dnf":
            output = run_cmd(["dnf", "update", "-y"], sudo=True)
        elif name == "pacman":
            output = run_cmd(["pacman", "-Syu", "--noconfirm"], sudo=True)
        elif name == "brew":
            output = run_cmd(["brew", "upgrade"])
        else:
            output = f"Upgrade not implemented for {name}"
        
        print(output)

    elif cmd == "search":
        if not args:
            print("ERROR: Usage: package.py search <query>")
            return 1
        
        name, _ = detect_package_manager()
        if not name:
            print("ERROR: No supported package manager found")
            return 1
        
        query = " ".join(args)
        if name == "apt":
            output = run_cmd(["apt-cache", "search", query])
        elif name == "yum":
            output = run_cmd(["yum", "search", query])
        elif name == "dnf":
            output = run_cmd(["dnf", "search", query])
        elif name == "pacman":
            output = run_cmd(["pacman", "-Ss", query])
        elif name == "brew":
            output = run_cmd(["brew", "search", query])
        else:
            output = f"Search not implemented for {name}"
        
        print(output)

    elif cmd == "list":
        name, _ = detect_package_manager()
        if not name:
            print("ERROR: No supported package manager found")
            return 1
        
        if name == "apt":
            output = run_cmd(["dpkg", "--list"])
        elif name == "yum":
            output = run_cmd(["yum", "list", "installed"])
        elif name == "dnf":
            output = run_cmd(["dnf", "list", "installed"])
        elif name == "pacman":
            output = run_cmd(["pacman", "-Q"])
        elif name == "brew":
            output = run_cmd(["brew", "list"])
        else:
            output = f"List not implemented for {name}"
        
        print(output)

    else:
        print(f"ERROR: Unknown command: {cmd}")
        print(__doc__)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
