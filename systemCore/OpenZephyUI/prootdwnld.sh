#!/bin/bash

# Script to download and set up Ubuntu Proot-Distro (amd64)

# --- Configuration ---
DISTRO_NAME="ubuntu"
DISTRO_VERSION="latest" # Or specify a specific version like "22.04"
ROOTFS_NAME="${DISTRO_NAME}-${DISTRO_VERSION}-proot-distro"
ARCH="amd64"
INSTALL_DIR="${HOME}/.proot-distro/${ROOTFS_NAME}"

# --- Functions ---

# Function to check for required commands
check_command() {
  if ! command -v "$1" &> /dev/null; then
    echo "Error: $1 is not installed. Please install it and try again."
    exit 1
  fi
}

# Function to download the rootfs
download_rootfs() {
  local url="$1"
  local filename="$2"
  echo "Downloading $filename..."
  if ! wget -c "$url" -O "$filename"; then
      echo "Error: Failed to download $filename."
      echo "Please check your internet connection and try again. You may also try using curl instead of wget."
      echo "To use curl, comment the 'wget' line above and uncomment the 'curl' line below:"
      echo "# curl -L -C - -o \"$filename\" \"$url\""
      exit 1
  fi
  # Alternatively, you can use curl:
  # curl -L -C - -o "$filename" "$url"
}

# Function to extract the rootfs
extract_rootfs() {
  local filename="$1"
  local target_dir="$2"

  echo "Extracting $filename to $target_dir..."
  if ! tar -xf "$filename" -C "$target_dir"; then
    echo "Error: Failed to extract $filename."
    exit 1
  fi
}

# Function to check for root privileges (if needed)
check_root() {
  if [ "$EUID" -ne 0 ]; then
    echo "Error: This script requires root privileges for certain operations."
    echo "Please run with 'sudo $0' or consider using the --no-install option for proot-distro install later."
    exit 1
  fi
}

# --- Main Script ---

# Check for required commands
check_command "proot-distro"
check_command "wget"
check_command "tar"
check_command "gawk"
# Optional, but useful for installation
check_command "sudo" # Comment this line if you are running as root or are certain of the --no-install method

# Get the Ubuntu release information from proot-distro's repository
if [[ "$DISTRO_VERSION" == "latest" ]]; then
  DISTRO_VERSION=$(proot-distro list | gawk -F ' *│ *|│' '$2 == "ubuntu" { print $3 }' | tail -n 1)
  if [[ -z "$DISTRO_VERSION" ]]; then
      echo "Error: Could not determine the latest Ubuntu version from proot-distro."
      exit 1
  fi
  echo "Latest Ubuntu version found: $DISTRO_VERSION"
fi

ROOTFS_NAME="${DISTRO_NAME}-${DISTRO_VERSION}-proot-distro"
INSTALL_DIR="${HOME}/.proot-distro/${ROOTFS_NAME}"

# Download link from the official Proot-Distro GitHub releases
DOWNLOAD_URL="https://github.com/AndronixApp/AndronixOrigin/releases/latest/download/${DISTRO_NAME}-${DISTRO_VERSION}-${ARCH}.tar.xz"

# Download the rootfs
mkdir -p "$INSTALL_DIR"
download_rootfs "$DOWNLOAD_URL" "${DISTRO_NAME}-${DISTRO_VERSION}-${ARCH}.tar.xz"

# Extract the rootfs
extract_rootfs "${DISTRO_NAME}-${DISTRO_VERSION}-${ARCH}.tar.xz" "$INSTALL_DIR"

# Install the distro using proot-distro (requires root for some features)
#check_root # Uncomment this if you are not certain
echo "Installing Ubuntu with proot-distro..."
if ! sudo proot-distro install --no-install --rootfs "${INSTALL_DIR}" "$DISTRO_NAME" "$DISTRO_VERSION"; then
  echo "Error: Failed to install Ubuntu with proot-distro."
  echo "You may need to install it manually or run this script as root."
  echo "You may try using proot-distro install --rootfs \"${INSTALL_DIR}\" \"$DISTRO_NAME\" \"$DISTRO_VERSION\""
  echo "Or, with root privileges: sudo proot-distro install --rootfs \"${INSTALL_DIR}\" \"$DISTRO_NAME\" \"$DISTRO_VERSION\""
  exit 1
fi

# Clean up the downloaded archive
rm "${DISTRO_NAME}-${DISTRO_VERSION}-${ARCH}.tar.xz"

# Success message
echo "Ubuntu ($DISTRO_VERSION, $ARCH) Proot-Distro has been downloaded and installed successfully!"
echo "You can now start it with: proot-distro login $DISTRO_NAME"
echo "Rootfs is located at: $INSTALL_DIR"