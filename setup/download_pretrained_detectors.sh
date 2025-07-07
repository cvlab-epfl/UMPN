#!/bin/bash

# UMPN Pretrained Detectors Download Script
# Downloads the pretrained detectors archive from the online repository

# Configuration
BASE_URL="https://scout.epfl.ch/datas/umpn_data/pretrained_detectors"
PRETRAINED_DIR="pretrained_detectors"
ARCHIVE_NAME="pretrained_detectors.tar.gz"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}UMPN Pretrained Detectors Download Script${NC}"
echo "==========================================="

# Check if pretrained_detectors directory already exists
if [ -d "$PRETRAINED_DIR" ]; then
    echo -e "${YELLOW}⚠ Pretrained detectors directory already exists${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing directory...${NC}"
        rm -rf "$PRETRAINED_DIR"
        echo -e "${GREEN}✓ Existing directory removed${NC}"
    else
        echo -e "${BLUE}Skipping extraction. Exiting...${NC}"
        exit 0
    fi
fi

# Check if archive already exists
if [ -f "$ARCHIVE_NAME" ]; then
    echo -e "${YELLOW}⚠ ${ARCHIVE_NAME} already exists in current directory.${NC}"
    read -p "Do you want to re-download it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Skipping download. Proceeding with extraction...${NC}"
        SKIP_DOWNLOAD=true
    else
        echo -e "${YELLOW}Removing existing archive...${NC}"
        rm -f "$ARCHIVE_NAME"
        SKIP_DOWNLOAD=false
    fi
else
    SKIP_DOWNLOAD=false
fi

# Download the archive if it doesn't exist
if [ ! -f "$ARCHIVE_NAME" ] && [ "$SKIP_DOWNLOAD" != true ]; then
    echo ""
    echo -e "${YELLOW}Downloading ${ARCHIVE_NAME}...${NC}"
    echo -e "${BLUE}Source: ${BASE_URL}/${ARCHIVE_NAME}${NC}"
    
    # Download the file
    if curl -L -o "$ARCHIVE_NAME" "$BASE_URL/$ARCHIVE_NAME"; then
        echo -e "${GREEN}✓ ${ARCHIVE_NAME} downloaded successfully${NC}"
        
        # Verify the file was downloaded and has content
        if [ -s "$ARCHIVE_NAME" ]; then
            file_size=$(ls -lh "$ARCHIVE_NAME" | awk '{print $5}')
            echo -e "${GREEN}✓ Archive size: ${file_size}${NC}"
        else
            echo -e "${RED}✗ Downloaded file is empty. Removing...${NC}"
            rm -f "$ARCHIVE_NAME"
            exit 1
        fi
    else
        echo -e "${RED}✗ Failed to download ${ARCHIVE_NAME}${NC}"
        exit 1
    fi
fi

# Extract the archive
echo ""
echo -e "${YELLOW}Extracting ${ARCHIVE_NAME}...${NC}"

if tar -xzf "$ARCHIVE_NAME"; then
    echo -e "${GREEN}✓ Archive extracted successfully${NC}"
    
    # Verify the directory was created
    if [ -d "$PRETRAINED_DIR" ]; then
        echo -e "${GREEN}✓ Pretrained detectors directory created${NC}"
    else
        echo -e "${RED}✗ Pretrained detectors directory not found after extraction${NC}"
        exit 1
    fi
    
    # Clean up the downloaded archive
    echo -e "${YELLOW}Cleaning up downloaded archive...${NC}"
    rm -f "$ARCHIVE_NAME"
    echo -e "${GREEN}✓ Archive removed${NC}"
else
    echo -e "${RED}✗ Failed to extract ${ARCHIVE_NAME}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}==========================================="
echo -e "✓ Pretrained detectors setup complete!"
echo -e "==========================================${NC}"

# List the contents of the pretrained_detectors directory
echo ""
echo "Contents of pretrained_detectors directory:"
if [ -d "$PRETRAINED_DIR" ]; then
    ls -lh "$PRETRAINED_DIR"/
else
    echo -e "${RED}✗ Pretrained detectors directory not found${NC}"
fi

echo ""
echo -e "${GREEN}Setup complete! You can now use the pretrained detectors for training and evaluation.${NC}"
