#!/bin/bash

# Check if usr folder exists
if [ ! -d "usr" ]; then
    echo "Error: 'usr' folder not found. Make sure you are in the root directory of the project."
    echo "Usage: bash ./builder-tool/maclauncherbuilder.sh"
    exit 1
fi

# Check if launchcontrol folder exists
if [ ! -d "launchcontrol" ]; then
    echo "Error: 'launchcontrol' folder not found. Make sure you are in the root directory of the project."
    echo "Usage: bash ./builder-tool/maclauncherbuilder.sh"
    exit 1
fi

# Define variables
APP_NAME="Adelaide Zephyrine Charlotte"
AUTHOR="Albert Starfield Wahyu Suryo Samudro"
ICON_PATH="$(pwd)/usr/icon/mac/icon.icns"
BUNDLE_DIR="$(pwd)/builder-tool/autoEasySetup"
APP_DIR="$BUNDLE_DIR/$APP_NAME.app"
LAUNCH_SCRIPT="$HOME/adelaide-zephyrine-charlotte-assistant/launchcontrol/run.sh"

# Create app bundle directory
mkdir -p "$APP_DIR/Contents/MacOS"

# Create Info.plist file
cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.$APP_NAME</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>NSAppleScriptEnabled</key>
    <true/>
</dict>
</plist>
EOF

# Copy icon file
cp "$ICON_PATH" "$APP_DIR/Contents/Resources/icon.icns"

# Create launch script
cat > "$APP_DIR/Contents/MacOS/$APP_NAME" <<EOF
#!/bin/bash

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    /bin/bash -c "\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Update Homebrew
brew update

# Upgrade packages
brew upgrade -y

# Install git if not installed
if ! command -v git &> /dev/null; then
    brew install git -y
fi

# Clone repository if run.sh not detected
if [ ! -f "$LAUNCH_SCRIPT" ]; then
    git clone https://github.com/albertstarfield/alpaca-electron-zephyrine "$HOME/adelaide-zephyrine-charlotte-assistant"
fi

# Launch terminal and execute run.sh
osascript -e 'tell application "Terminal" to do script "bash $LAUNCH_SCRIPT"'

# Wait for 15 seconds
sleep 15
EOF

# Make launch script executable
chmod +x "$APP_DIR/Contents/MacOS/$APP_NAME"

echo "App bundle created at: $APP_DIR"
