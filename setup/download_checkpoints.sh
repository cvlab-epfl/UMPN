#!/bin/bash

# UMPN Model Checkpoints Download Script
# Downloads the three model checkpoints from the online repository

# Configuration
BASE_URL="https://scout.epfl.ch/datas/umpn_data/model_checkpoints"
WEIGHTS_DIR="weights"
CHECKPOINTS=("best_mot17.pth.tar" "best_mot20.pth.tar" "best_wildtrack.pth.tar")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}UMPN Model Checkpoints Download Script${NC}"
echo "======================================"

# Create weights directory if it doesn't exist
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo -e "${YELLOW}Creating weights directory...${NC}"
    mkdir -p "$WEIGHTS_DIR"
    echo -e "${GREEN}✓ Weights directory created${NC}"
else
    echo -e "${GREEN}✓ Weights directory already exists${NC}"
fi

# Download each checkpoint
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo ""
    echo -e "${YELLOW}Downloading ${checkpoint}...${NC}"
    
    # Check if file already exists
    if [ -f "$WEIGHTS_DIR/$checkpoint" ]; then
        echo -e "${YELLOW}⚠ ${checkpoint} already exists. Skipping download.${NC}"
        continue
    fi
    
    # Download the file
    if curl -L -o "$WEIGHTS_DIR/$checkpoint" "$BASE_URL/$checkpoint"; then
        echo -e "${GREEN}✓ ${checkpoint} downloaded successfully${NC}"
        
        # Verify the file was downloaded and has content
        if [ -s "$WEIGHTS_DIR/$checkpoint" ]; then
            file_size=$(ls -lh "$WEIGHTS_DIR/$checkpoint" | awk '{print $5}')
            echo -e "${GREEN}✓ File size: ${file_size}${NC}"
        else
            echo -e "${RED}✗ Downloaded file is empty. Removing...${NC}"
            rm -f "$WEIGHTS_DIR/$checkpoint"
            exit 1
        fi
    else
        echo -e "${RED}✗ Failed to download ${checkpoint}${NC}"
        exit 1
    fi
done

echo ""
echo -e "${GREEN}======================================"
echo -e "✓ All checkpoints downloaded successfully!"
echo -e "=====================================${NC}"

# List the downloaded files
echo ""
echo "Downloaded checkpoints:"
ls -lh "$WEIGHTS_DIR"/*.pth.tar

echo ""
echo -e "${GREEN}Setup complete! You can now use the model checkpoints for training and evaluation.${NC}"
