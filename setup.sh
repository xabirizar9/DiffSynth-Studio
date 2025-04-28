#!/bin/bash

# Install zsh and Oh My Zsh
echo "Installing zsh and Oh My Zsh..."
sudo apt-get update
sudo apt-get install -y zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Create directory for CelebV-HQ dataset
echo "Creating directory for CelebV-HQ dataset..."
sudo mkdir -p /mnt/disks/celebv-hq

# Find and mount the CelebV-HQ disk
echo "Finding and mounting CelebV-HQ disk..."
DISK_ID=$(ls -l /dev/disk/by-id/google* | grep "google-celebv-hq" | awk '{print $NF}' | sed 's/\.\.\/\.\.\///')
if [[ $DISK_ID == *"sda"* ]]; then
  DISK_PATH="/dev/sda"
elif [[ $DISK_ID == *"sdb"* ]]; then
  DISK_PATH="/dev/sdb"
else
  echo "Error: Could not find the CelebV-HQ disk"
  exit 1
fi

echo "Mounting disk $DISK_PATH to /mnt/disks/celebv-hq..."
sudo mount -o discard,defaults $DISK_PATH /mnt/disks/celebv-hq/

# Create symbolic link to dataset
echo "Creating symbolic link to dataset..."
ln -s /mnt/disks/celebv-hq/CelebV-HQ ./dataset

sudo chmod -R 777 /mnt/disks/celebv-hq/

# Install uv and sync dependencies
echo "Installing uv and syncing dependencies..."
pip install uv
uv sync

# Download model from Hugging Face
echo "Downloading model from Hugging Face..."
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan-AI/Wan2.1-T2V-1.3B

# Create virtual environment activation command
echo "Setting up virtual environment..."
python -m venv .venv

# Print instructions for activating the virtual environment
echo ""
echo "======================================================"
echo "Setup complete! To activate the virtual environment, run:"
echo "source .venv/bin/activate"
echo "======================================================"
