#!/bin/bash

# Check if the necessary directories exist
if [ ! -d "usr" ] || [ ! -d "launchcontrol" ]; then
    echo "Error: Make sure you are on the root directory of the project and you can launch like bash ./builder-tool/macOSdarwinBundle/maclauncherbuilder.sh"
    exit 1
fi

echo "Bruh Linux is just as easy as launching it through the terminal and all is done! :D"
cp  ./builder-tool/LinuxSpecificInstaller/exec_L0 ./builder-tool/autoEasySetup/adelaide-zephyrine-charlotte