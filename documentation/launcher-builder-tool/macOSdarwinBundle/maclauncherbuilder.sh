#!/bin/bash

# Check if the necessary directories exist
if [ ! -d "usr" ] || [ ! -d "launchcontrol" ]; then
    echo "Error: Make sure you are on the root directory of the project and you can launch like bash ./builder-tool/macOSdarwinBundle/maclauncherbuilder.sh"
    exit 1
fi

# Create the .app directory structure
mkdir -p "./builder-tool/autoEasySetup/Adelaide Zephyrine Charlotte.app/Contents/MacOS"
mkdir -p "./builder-tool/autoEasySetup/Adelaide Zephyrine Charlotte.app/Contents/Resources"

# Copy the icon file
cp "usr/icon/mac/icon.icns" "./builder-tool/autoEasySetup/Adelaide Zephyrine Charlotte.app/Contents/Resources/"

# Create the Info.plist file
cat > "./builder-tool/autoEasySetup/Adelaide Zephyrine Charlotte.app/Contents/Info.plist" <<EOL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>Adelaide Zephyrine Charlotte</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.Adelaide-Zephyrine-Charlotte</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Adelaide Zephyrine Charlotte</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
</dict>
</plist>
EOL

# Create the executable bash script
cat ./builder-tool/macOSdarwinBundle/exec_L0 > "./builder-tool/autoEasySetup/Adelaide Zephyrine Charlotte.app/Contents/MacOS/Adelaide Zephyrine Charlotte"


# Make the script executable
chmod +x "./builder-tool/autoEasySetup/Adelaide Zephyrine Charlotte.app/Contents/MacOS/Adelaide Zephyrine Charlotte"

echo "App assembled successfully!"
