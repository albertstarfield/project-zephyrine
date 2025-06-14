#!/bin/bash

# create_and_run_proxy.sh
# Takes one argument: the path to the temporary AppleScript file

set -e # Exit on errors

TEMP_APPLESCRIPT_PATH="$1"
TEMP_APP_NAME="HelperApp_$(uuidgen)" # Create unique name
TEMP_APP_DIR="/tmp/${TEMP_APP_NAME}.app" # Create in /tmp for simplicity
MACOS_DIR="${TEMP_APP_DIR}/Contents/MacOS"
RES_DIR="${TEMP_APP_DIR}/Contents/Resources/Scripts"
EXECUTABLE_NAME="${TEMP_APP_NAME}" # Match app name

echo "Helper Script: Received AppleScript path: ${TEMP_APPLESCRIPT_PATH}"
echo "Helper Script: Creating temporary app at: ${TEMP_APP_DIR}"

# --- Cleanup trap ---
cleanup() {
  echo "Helper Script: Cleaning up ${TEMP_APP_DIR} and ${TEMP_APPLESCRIPT_PATH}..."
  rm -rf "${TEMP_APP_DIR}"
  # rm -f "${TEMP_APPLESCRIPT_PATH}" # Python script should delete this one
  echo "Helper Script: Cleanup finished."
}
trap cleanup EXIT SIGINT SIGTERM

# --- Create App Structure ---
mkdir -p "${MACOS_DIR}"
mkdir -p "${RES_DIR}"

# --- Create Info.plist ---
cat > "${TEMP_APP_DIR}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key>
	<string>${EXECUTABLE_NAME}</string>
	<key>CFBundleIdentifier</key>
	<string>com.amaryllis.temp.${TEMP_APP_NAME}</string> <!-- Unique ID -->
	<key>CFBundleName</key>
	<string>${TEMP_APP_NAME}</string>
	<key>CFBundleVersion</key>
	<string>1.0</string>
    <key>LSUIElement</key> <!-- Optional: Makes it run without Dock icon -->
    <true/>
</dict>
</plist>
EOF

# --- Copy AppleScript ---
cp "${TEMP_APPLESCRIPT_PATH}" "${RES_DIR}/main.scpt"

# --- Create Executable Stub ---
cat > "${MACOS_DIR}/${EXECUTABLE_NAME}" << EOF
#!/bin/bash
# Get the directory containing this script (MacOS inside .app)
DIR=\$(cd "\$(dirname "\$0")" && pwd)
# Path to the AppleScript relative to the executable
SCRIPT_PATH="\${DIR}/../Resources/Scripts/main.scpt"
# Execute the AppleScript and echo its output
/usr/bin/osascript "\${SCRIPT_PATH}"
EOF
chmod +x "${MACOS_DIR}/${EXECUTABLE_NAME}"

# --- (Optional) Code Signing ---
# DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)" # Replace with your actual ID
# if [[ -n "$DEVELOPER_ID" ]]; then
#   echo "Helper Script: Attempting to sign the app..."
#   codesign --force --deep --sign "$DEVELOPER_ID" "${TEMP_APP_DIR}" || echo "Warning: Codesigning failed."
# else
#   echo "Helper Script: Developer ID not set, skipping codesigning."
# fi

# --- Execute the temporary app's script ---
echo "Helper Script: Executing the action via osascript..."
# Execute the main AppleScript, output goes to stdout/stderr of this script
osascript "${RES_DIR}/main.scpt"

EXEC_RESULT=$?
echo "Helper Script: osascript finished with code ${EXEC_RESULT}"

exit $EXEC_RESULT # Return the exit code of osascript