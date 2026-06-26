#!/usr/bin/env python3
"""
Create macOS .app bundle for Adelaide with proper permissions.

This script creates a minimal .app bundle with Info.plist containing
NSMicrophoneUsageDescription, NSCameraUsageDescription, and
NSScreenCaptureUsageDescription for microphone, camera, and screen capture access.

Usage:
    python3 create_macos_app.py [--output Adelaide.app]

The resulting .app bundle can be launched directly or used with py2app.
"""

import os
import sys
import argparse
import stat
from pathlib import Path


INFO_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Adelaide</string>
    <key>CFBundleDisplayName</key>
    <string>Adelaide Zephyrine Assistant</string>
    <key>CFBundleIdentifier</key>
    <string>com.zephyrine.adelaide</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>

    <!-- Privacy: Microphone Access -->
    <key>NSMicrophoneUsageDescription</key>
    <string>Adelaide needs microphone access for voice interaction and speech recognition.</string>

    <!-- Privacy: Camera Access -->
    <key>NSCameraUsageDescription</key>
    <string>Adelaide needs camera access for visual context and multimodal interaction.</string>

    <!-- Privacy: Screen Capture Access (macOS 10.15+) -->
    <key>NSScreenCaptureUsageDescription</key>
    <string>Adelaide needs screen capture access to understand visual context from your screen.</string>

    <!-- Privacy: File Access -->
    <key>NSDocumentsFolderUsageDescription</key>
    <string>Adelaide needs access to your documents folder for file operations.</string>

    <!-- Privacy: Downloads Access -->
    <key>NSDownloadsFolderUsageDescription</key>
    <string>Adelaide needs access to your downloads folder for file operations.</string>
</dict>
</plist>
"""


LAUNCHER_TEMPLATE = """#!/bin/bash
# Adelaide Launcher Script
# This script launches Adelaide using the venv Python interpreter

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
ADelaide_DIR="${APP_DIR}/../../../.."

# Find venv Python
VENV_PYTHON="${ADelaide_DIR}/pyvenv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="${ADelaide_DIR}/pyvenv/Scripts/python.exe"
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Venv Python not found. Using system Python."
    VENV_PYTHON="python3"
fi

# Launch sidecar UI
cd "${ADelaide_DIR}/ui"
exec "$VENV_PYTHON" sidecar_ui.py "$@"
"""


def create_app_bundle(output_path: str) -> None:
    """Create macOS .app bundle with permissions."""
    app_path = Path(output_path)
    
    # Create directory structure
    contents_dir = app_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Write Info.plist
    plist_path = contents_dir / "Info.plist"
    with open(plist_path, "w") as f:
        f.write(INFO_PLIST_TEMPLATE)
    print(f"[+] Created Info.plist at {plist_path}")
    
    # Write launcher script
    launcher_path = macos_dir / "launcher"
    with open(launcher_path, "w") as f:
        f.write(LAUNCHER_TEMPLATE)
    
    # Make launcher executable
    launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[+] Created launcher at {launcher_path}")
    
    print(f"\n[+] App bundle created at: {app_path}")
    print(f"    Double-click to launch, or run:")
    print(f"    open {app_path}")


def main():
    parser = argparse.ArgumentParser(description="Create macOS .app bundle for Adelaide")
    parser.add_argument(
        "--output", "-o",
        default="Adelaide.app",
        help="Output path for .app bundle (default: Adelaide.app)"
    )
    args = parser.parse_args()
    
    create_app_bundle(args.output)


if __name__ == "__main__":
    main()
