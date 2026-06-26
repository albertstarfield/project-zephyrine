#!/usr/bin/env python3
"""
Create macOS .app bundle for Adelaide Zephyrine Assistant.

This script creates a minimal .app bundle with:
- Info.plist with microphone/camera/screen capture permissions
- Launcher script that opens Terminal and runs the server with GUI
- Auto-installs to /Applications on first run
- Ad-hoc code signing for Gatekeeper compatibility

Usage:
    python3 create_macos_app.py [--output Adelaide Zephyrine Assistant.app]

Code Signing Options:
- Ad-hoc (default): No Developer ID required, prevents Gatekeeper warning
- Developer ID: Requires Apple Developer account ($99/year)
- Notarization: Requires Developer ID + notarization via Apple

For distribution outside App Store:
1. Get Apple Developer account
2. Create Developer ID Application certificate
3. Sign with: codesign --force --deep --sign "Developer ID Application: Your Name (TEAM_ID)" "Adelaide Zephyrine Assistant.app"
4. Notarize with: xcrun notarytool submit "Adelaide Zephyrine Assistant.app" --apple-id your@email.com --team-id TEAM_ID
"""

import os
import argparse
import stat
import subprocess
from pathlib import Path


INFO_PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Adelaide Zephyrine Assistant</string>
    <key>CFBundleDisplayName</key>
    <string>Adelaide Zephyrine Assistant</string>
    <key>CFBundleIdentifier</key>
    <string>com.zephyrine.adelaide-assistant</string>
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
# Adelaide Zephyrine Assistant Launcher
# This script opens Terminal and runs the server with GUI

# Get the directory where this .app is located
APP_DIR="$(dirname "$(dirname "$0")")"

# [DO NOT REMOVE] Use BASE_DIR/run for temp files, not /tmp
# This ensures temp files are in the project directory, not system temp
RUN_DIR="$HOME/LibraryTube/OpenIntellegentiaPlatform/Adelaide_Lite/run"
mkdir -p "$RUN_DIR"

# Try to find Adelaide_Lite directory
# Check common locations relative to .app
SEARCH_DIRS=(
    "$APP_DIR"
    "$HOME/LibraryTube/OpenIntellegentiaPlatform/Adelaide_Lite"
    "$HOME/OpenIntellegentiaPlatform/Adelaide_Lite"
    "$HOME/Adelaide_Lite"
    "$HOME/Desktop/Adelaide_Lite"
    "$HOME/Documents/Adelaide_Lite"
)

ADelaide_DIR=""
for dir in "${SEARCH_DIRS[@]}"; do
    if [ -f "$dir/run.py" ]; then
        ADelaide_DIR="$dir"
        break
    fi
done

if [ -z "$ADelaide_DIR" ]; then
    # Ask user to select directory, store temp file in BASE_DIR/run
    TEMP_FILE="$RUN_DIR/adelaide_dir_select.txt"
    osascript -e 'tell application "Finder"
        set dir to POSIX path of (choose folder with prompt "Select Adelaide_Lite directory")
        return dir
    end tell' > "$TEMP_FILE"
    ADelaide_DIR=$(cat "$TEMP_FILE" | tr -d '\n')
    rm -f "$TEMP_FILE"
fi

if [ -z "$ADelaide_DIR" ]; then
    osascript -e 'display dialog "Could not find Adelaide_Lite directory." buttons {"OK"} default button 1'
    exit 1
fi

# Open Terminal and run the server with GUI
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \\"$ADelaide_DIR\\" && python3 run.py"
end tell
EOF
"""


def create_app_bundle(output_path: str) -> None:
    """Create macOS .app bundle with permissions and launcher."""
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
    
    # [DO NOT REMOVE] Ad-hoc code signing for macOS Gatekeeper
    # Sign the app bundle with ad-hoc signature (no Developer ID required)
    # This prevents Gatekeeper from blocking the app on launch
    # For distribution, you'll need a proper Developer ID certificate
    try:
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(app_path)],
            check=True,
            capture_output=True
        )
        print("[+] Signed app bundle with ad-hoc signature")
    except subprocess.CalledProcessError as e:
        print(f"[!] Warning: Could not sign app bundle: {e}")
        print("    App may show Gatekeeper warning on first launch")
    
    print(f"\n[+] App bundle created at: {app_path}")
    print("    Double-click to launch, or run:")
    print(f'    open "{app_path}"')


def install_to_applications(app_path: str) -> str:
    """Install .app bundle to /Applications."""
    app_name = os.path.basename(app_path)
    applications_dir = "/Applications"
    dest_path = os.path.join(applications_dir, app_name)
    
    # Check if already installed
    if os.path.exists(dest_path):
        print(f"[*] App already installed at {dest_path}")
        return dest_path
    
    # Copy to /Applications
    try:
        subprocess.run(["cp", "-R", app_path, dest_path], check=True)
        print(f"[+] Installed to {dest_path}")
        return dest_path
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to install to /Applications: {e}")
        print("    You may need to run with sudo or drag manually to Applications")
        return app_path


def main():
    parser = argparse.ArgumentParser(description="Create macOS .app bundle for Adelaide")
    parser.add_argument(
        "--output", "-o",
        default="Adelaide Zephyrine Assistant.app",
        help="Output path for .app bundle (default: Adelaide Zephyrine Assistant.app)"
    )
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Install to /Applications after creating"
    )
    args = parser.parse_args()
    
    create_app_bundle(args.output)
    
    if args.install:
        install_to_applications(args.output)


if __name__ == "__main__":
    main()
