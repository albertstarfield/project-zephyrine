#!/bin/bash

# Define source image and target directory
SOURCE_IMG="public/img/AdelaideEntity.png"
TARGET_DIR="public/img"

# --- Safety Check ---
if [ ! -f "$SOURCE_IMG" ]; then
  echo "Error: Source image not found at '$SOURCE_IMG'"
  echo "Please ensure the image exists and you are in the 'systemCore/frontend-face-zephyrine' directory."
  exit 1
fi

echo "Generating PWA icons in '$TARGET_DIR' from '$SOURCE_IMG'..."

# --- Generate Standard Icons ---

# Generate 192x192 icon (common PWA size)
echo "Creating 192x192 icon..."
convert "$SOURCE_IMG" -resize 192x192! "${TARGET_DIR}/AdelaideEntity_192.png"
# The '!' forces exact dimensions, ignoring aspect ratio if necessary for square icons.
# Remove '!' if you want to preserve aspect ratio and potentially have padding.

# Generate 512x512 icon (larger PWA size)
echo "Creating 512x512 icon..."
convert "$SOURCE_IMG" -resize 512x512! "${TARGET_DIR}/AdelaideEntity_512.png"

# Generate 180x180 icon (common Apple Touch Icon size)
echo "Creating 180x180 icon (Apple Touch)..."
convert "$SOURCE_IMG" -resize 180x180! "${TARGET_DIR}/AdelaideEntity_180.png"


# --- Generate Maskable Icon (512x512) ---
# A maskable icon needs the main content within a "safe zone" (approx central 80%)
# This command resizes the original to fit within the safe zone (~410px for 512px canvas),
# then centers it on a transparent 512x512 canvas.
SAFE_ZONE_SIZE=410 # Approx 80% of 512
echo "Creating 512x512 maskable icon (content within ~${SAFE_ZONE_SIZE}px safe zone)..."
convert "$SOURCE_IMG" -resize ${SAFE_ZONE_SIZE}x${SAFE_ZONE_SIZE} \
          -background none -gravity center -extent 512x512 \
          "${TARGET_DIR}/AdelaideEntity_maskable.png"

echo "---"
echo "Icon generation complete!"
echo "Generated files in '$TARGET_DIR':"
echo "- AdelaideEntity_192.png"
echo "- AdelaideEntity_512.png"
echo "- AdelaideEntity_180.png"
echo "- AdelaideEntity_maskable.png"
echo "Make sure these paths match your vite.config.js manifest settings."
echo "You may want to visually inspect the maskable icon to ensure it looks correct."