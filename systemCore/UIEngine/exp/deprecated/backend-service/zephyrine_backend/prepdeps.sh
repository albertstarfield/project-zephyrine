#!/bin/bash
# The final, correct script. It vendors Alire dependencies by copying
# the unpacked source directories directly, as discovered on the user's system.

#set -e # Stop the script if any command fails.

# --- Step 1: Add and Download Dependencies ---
echo "--- Step 1: Configuring and updating dependencies ---"
alr -n with aws gnatcoll libgpr xmlada
alr update
echo ""

# --- Step 2: Discover the Alire Releases Directory ---
RELEASES_DIR=""
path1="$HOME/.alire/releases"
path2="$HOME/.local/share/alire/releases"

echo "--- Step 2: Locating Alire's Release Directory ---"
if [ -d "$path1" ]; then
    RELEASES_DIR="$path1"
elif [ -d "$path2" ]; then
    RELEASES_DIR="$path2"
else
    echo "ERROR: Could not find Alire 'releases' directory." >&2
    exit 1
fi
echo "Found release directory: $RELEASES_DIR"
echo ""

# --- Step 3: Vendor Dependencies by Copying Directories ---
DEPS_DIR="alire/deps"
mkdir -p "$DEPS_DIR"

# Helper function to find and copy a dependency directory.
copy_dep_dir() {
    local dep_name=$1
    echo "--> Processing dependency: $dep_name"

    # Find the source directory for the dependency.
    # The -print -quit makes it stop after the first match.
    # The -type d ensures we only find directories.
    local src_dir=$(find "$RELEASES_DIR" -name "${dep_name}_*" -type d -print -quit)

    if [ -n "$src_dir" ]; then
        echo "    Found source directory: $src_dir"
        # Copy the entire directory recursively into our local deps folder.
        cp -R "$src_dir" "$DEPS_DIR/"
        echo "    Successfully copied $dep_name."
    else
        echo "    ERROR: Could not find source directory for $dep_name in $RELEASES_DIR" >&2
    fi
    echo ""
}

echo "--- Step 3: Vendoring dependencies into ./alire/deps/ ---"
copy_dep_dir "aws"
copy_dep_dir "gnatcoll"
copy_dep_dir "libgpr"
copy_dep_dir "xmlada"

echo "--- Project setup and vendoring complete. ---"
echo "Final contents of alire/deps:"
ls -l "$DEPS_DIR"