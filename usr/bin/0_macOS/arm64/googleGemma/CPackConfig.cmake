# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp;/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Taku Kudo")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/opt/homebrew/Cellar/cmake/3.29.0/share/cmake/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "gemma built using CMake")
set(CPACK_DMG_SLA_USE_RESOURCE_FILE_LICENSE "ON")
set(CPACK_GENERATOR "7Z")
set(CPACK_INNOSETUP_ARCHITECTURE "x64")
set(CPACK_INSTALL_CMAKE_PROJECTS "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build;gemma;ALL;/")
set(CPACK_INSTALL_PREFIX "/usr/local")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "gemma 0.2.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "gemma 0.2.0")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJDUMP_EXECUTABLE "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/objdump")
set(CPACK_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.4.sdk")
set(CPACK_OUTPUT_CONFIG_FILE "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/CPackConfig.cmake")
set(CPACK_PACKAGE_CONTACT "taku@google.com")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/opt/homebrew/Cellar/cmake/3.29.0/share/cmake/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "gemma built using CMake")
set(CPACK_PACKAGE_FILE_NAME "gemma-0.2.0-Darwin")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "gemma 0.2.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "gemma 0.2.0")
set(CPACK_PACKAGE_NAME "gemma")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Humanity")
set(CPACK_PACKAGE_VERSION "0.2.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_RESOURCE_FILE_LICENSE "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src/LICENSE")
set(CPACK_RESOURCE_FILE_README "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/_deps/sentencepiece-src/README.md")
set(CPACK_RESOURCE_FILE_WELCOME "/opt/homebrew/Cellar/cmake/3.29.0/share/cmake/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TXZ")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.git/;/dist/;/sdist/;~$;")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/CPackSourceConfig.cmake")
set(CPACK_STRIP_FILES "TRUE")
set(CPACK_SYSTEM_NAME "Darwin")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Darwin")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/Users/user/adelaide-zephyrine-charlotte-assistant/usr/vendor/gemma.cpp/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
