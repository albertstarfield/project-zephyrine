#!/bin/bash

# Convert JPEG to PNG with rounded edges and alpha background
convert input.png -resize 16x16 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 16x16.png
convert input.png -resize 32x32 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 32x32.png
convert input.png -resize 64x64 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 64x64.png
convert input.png -resize 128x128 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 128x128.png
convert input.png -resize 256x256 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 256x256.png
convert input.png -resize 512x512 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 512x512.png
convert input.png -resize 1024x1024 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 1024x1024.png


# Convert 256x256.png to .ico and .icns
convert 1024x1024.png -alpha set -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage icon.ico
magick convert 256x256.png -define icon:auto-resize=16,48,256 -compress zip icon.icns

echo "Icons generated successfully"
