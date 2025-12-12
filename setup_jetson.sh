#!/bin/bash
#
# Jetson Nano Setup Script
# Installs all dependencies and configures the system for parking detection
#

set -e  # Exit on error

echo "=========================================="
echo "Jetson Nano Setup for Parking Detection"
echo "=========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: This script is designed for NVIDIA Jetson devices"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    python3-opencv \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran

# Install CUDA and TensorRT (if not already installed)
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Please install JetPack SDK first."
    echo "Visit: https://developer.nvidia.com/embedded/jetpack"
    exit 1
else
    echo "✓ CUDA found: $(nvcc --version | grep release)"
fi

# Create virtual environment
echo "Creating Python virtual environment..."
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch for Jetson
echo "Installing PyTorch for Jetson Nano..."
# PyTorch 2.1.0 for Jetson (adjust version as needed)
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl -O torch-2.1.0-cp38-cp38-linux_aarch64.whl
pip install torch-2.1.0-cp38-cp38-linux_aarch64.whl
rm torch-2.1.0-cp38-cp38-linux_aarch64.whl

# Install torchvision
echo "Installing torchvision..."
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.16.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.0
python setup.py install --user
cd ..
rm -rf torchvision

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Install pynvml for GPU monitoring
pip install nvidia-ml-py

# Install colorlog for colored logging
pip install colorlog

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p violations
mkdir -p output

# Set permissions
chmod +x main.py

# Install systemd service
echo "Installing systemd service..."
sudo cp systemd/parking-detector.service /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Convert models to TensorRT:"
echo "   python tensorrt_converter.py --model yolov8n.pt --output models/yolov8n.engine"
echo "   python tensorrt_converter.py --model NumberPlateDetection.pt --output models/NumberPlateDetection.engine"
echo ""
echo "2. Configure Firebase credentials:"
echo "   - Place your firebase-credentials.json in the project directory"
echo "   - Update config.yaml with your settings"
echo ""
echo "3. Test the system:"
echo "   python main.py --config config.yaml"
echo ""
echo "4. Enable auto-start on boot:"
echo "   sudo systemctl enable parking-detector"
echo "   sudo systemctl start parking-detector"
echo ""
echo "5. View logs:"
echo "   journalctl -u parking-detector -f"
echo ""
