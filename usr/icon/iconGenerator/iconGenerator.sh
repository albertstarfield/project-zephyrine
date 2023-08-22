#!/bin/bash

# Convert JPEG to PNG with rounded edges and alpha background
convert input.jpg -resize 16x16 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 16x16.png
convert input.jpg -resize 32x32 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 32x32.png
convert input.jpg -resize 64x64 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 64x64.png
convert input.jpg -resize 128x128 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 128x128.png
convert input.jpg -resize 256x256 -alpha set -background none -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 256x256.png

# Convert 256x256.png to .ico and .icns
convert 256x256.png -resize 16x16 -alpha set -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 16x16.ico
convert 256x256.png -resize 16x16 -alpha set -bordercolor none -border 8 -compose DstOver \( +clone -shadow 2x1+1+1 \) +swap -background none -layers merge +repage 16x16.icns

echo "Icons generated successfully"
