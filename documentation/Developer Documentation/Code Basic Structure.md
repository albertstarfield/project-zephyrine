# Subfolder Documentation
This subfolder contains information about the programs structure and how it was built using JavaScript. The main programs structure is located inside the usr folder, which contains six subfolders:
1. **bin**: This folder contains binary files that are generated and compiled when the installation process is run.
2. **icon**: This folder stores the icon for the program.
3. **nodemodules**: This folder appears after installation and contains modules that are required by the program.
4. **src**: This folder contains the main GUI files for the program.
5. **storage**: This folder is used to store future features such as save and restore chat functionality.
6. **Vendor folder**: This folder contains submodules from the GitHub repo, which are used without any modifications during compilation.
There is also a Dockerfile file for docker setup, which is not adapted for this version of the program. Additionally, there is a Docker-compose file that is not adapted for this version. There is an electron-builder.yml file, which is currently not being used. The index.js file contains the main logic for the program, and there is also a package-lock.json file that appears after running the installation process.
To install the program, simply run the run.sh file from the root directory of the repository. 