# No Parking Detection System

A production-grade automated parking violation detection system optimized for NVIDIA Jetson Nano with TensorRT acceleration, OCR-based license plate recognition, and cloud database integration.

## Features

- **TensorRT Acceleration**: 3-5x faster inference on Jetson Nano using FP16 precision
- **Vehicle Detection**: Real-time detection of cars, motorcycles, buses, and trucks
- **License Plate Recognition**: Automatic OCR-based license plate text extraction
- **Cloud Integration**: Firestore database for violation records with offline queue support
- **Multi-Camera Support**: RTSP streams, CSI cameras, USB webcams, and video files
- **Performance Monitoring**: Real-time FPS, GPU/CPU usage, and memory tracking
- **Auto-Recovery**: Automatic camera reconnection and error handling
- **24/7 Operation**: Systemd service for continuous operation

## System Requirements

### Hardware
- **NVIDIA Jetson Nano** (4GB recommended)
- **Camera**: CSI camera, USB webcam, or RTSP IP camera
- **Storage**: 16GB+ microSD card (32GB recommended)
- **Power**: 5V 4A power supply

### Software
- **JetPack SDK** 4.6+ (includes CUDA, cuDNN, TensorRT)
- **Python** 3.8+
- **OpenCV** 4.5+

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/NoParkingDetection.git
cd NoParkingDetection
```

### 2. Run Setup Script
```bash
chmod +x setup_jetson.sh
./setup_jetson.sh
```

This will:
- Install system dependencies
- Create Python virtual environment
- Install PyTorch for Jetson
- Install all Python packages
- Create necessary directories
- Set up systemd service

### 3. Convert Models to TensorRT
```bash
source venv/bin/activate

# Convert vehicle detection model
python tensorrt_converter.py --model yolov8n.pt --output models/yolov8n.engine --precision fp16

# Convert license plate detection model
python tensorrt_converter.py --model NumberPlateDetection.pt --output models/NumberPlateDetection.engine --precision fp16
```

**Note**: Model conversion takes 10-15 minutes. This is a one-time process.

### 4. Configure Firebase (Optional)
If using Firestore integration:

1. Create a Firebase project at https://console.firebase.google.com
2. Enable Firestore Database and Cloud Storage
3. Download service account credentials JSON
4. Place credentials file in project directory as `firebase-credentials.json`
5. Update `config.yaml` with your storage bucket name

### 5. Configure System
Edit `config.yaml` to match your setup:

```yaml
camera:
  source: "csi"  # or "rtsp", "usb", "video"
  # ... other camera settings

detection:
  use_tensorrt: true
  # ... detection thresholds

firestore:
  enabled: true
  credentials_path: "firebase-credentials.json"
  storage_bucket: "your-project.appspot.com"
```

## Usage

### Testing
Run the system manually for testing:

```bash
source venv/bin/activate
python main.py --config config.yaml
```

Press `q` to quit (if display window is enabled).

### Production Deployment
Enable auto-start on boot:

```bash
sudo systemctl enable parking-detector
sudo systemctl start parking-detector
```

### Monitor System
```bash
# View live logs
journalctl -u parking-detector -f

# Check service status
sudo systemctl status parking-detector

# Restart service
sudo systemctl restart parking-detector
```

## Configuration

### Camera Sources

**Video File**:
```yaml
camera:
  source: "video"
  video_path: "parking_lot_video.mp4"
```

**RTSP Stream**:
```yaml
camera:
  source: "rtsp"
  rtsp_url: "rtsp://username:password@192.168.1.100:554/stream"
```

**USB Camera**:
```yaml
camera:
  source: "usb"
  usb_device: 0
```

**CSI Camera** (Jetson Nano):
```yaml
camera:
  source: "csi"
  width: 1280
  height: 720
  fps: 30
```

### Detection Parameters

```yaml
detection:
  vehicle_confidence: 0.5      # Detection confidence threshold
  plate_confidence: 0.6        # License plate confidence
  position_threshold: 20       # Pixels for stationary detection
  time_threshold: 10           # Seconds before violation
```

### OCR Settings

```yaml
ocr:
  enabled: true
  gpu: true
  languages: ['en']            # Add more: ['en', 'hi', 'ar']
  confidence_threshold: 0.7
```

## Project Structure

```
NoParkingDetection/
├── main.py                      # Main entry point
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
│
├── tensorrt_converter.py        # Model conversion utility
├── tensorrt_detector.py         # TensorRT inference
├── ocr_engine.py               # OCR with EasyOCR
├── firestore_manager.py        # Cloud database
├── camera_manager.py           # Camera input handling
├── performance_monitor.py      # System monitoring
├── centroid_tracker.py         # Object tracking
├── models.py                   # Data models
│
├── setup_jetson.sh             # Installation script
├── systemd/                    # Systemd service files
├── logs/                       # Log files
├── models/                     # TensorRT engines
├── violations/                 # Local violation storage
└── output/                     # Output videos
```

## Performance

### Jetson Nano Benchmarks
- **FPS**: 15-20 FPS with TensorRT (FP16)
- **Inference Time**: 50-70ms per frame
- **Memory Usage**: 2-3GB RAM
- **GPU Utilization**: 70-90%

### Optimization Tips
1. Use TensorRT engines (3-5x speedup)
2. Enable FP16 precision
3. Reduce input resolution if needed
4. Disable display window for headless operation
5. Adjust `max_fps` in config to limit processing

## Firestore Data Structure

```
parking_violations/
  └── VIO_20231212_143052_1234/
      ├── violation_id: "VIO_20231212_143052_1234"
      ├── timestamp: "2023-12-12T14:30:52"
      ├── vehicle_id: 1234
      ├── license_plate: "ABC1234"
      ├── confidence: 0.85
      ├── parking_duration: 15.3
      ├── location: "Zone A"
      ├── vehicle_image_url: "https://..."
      └── plate_image_url: "https://..."
```

## Troubleshooting

### Camera Not Opening
- Check camera connection
- Verify camera source in config
- Test camera with: `v4l2-ctl --list-devices` (USB) or `gst-launch-1.0` (CSI)

### Low FPS
- Ensure TensorRT engines are being used (check logs)
- Reduce camera resolution
- Disable OCR temporarily to isolate bottleneck
- Check GPU usage: `tegrastats`

### TensorRT Conversion Fails
- Verify CUDA and TensorRT installation
- Check available disk space (needs ~2GB temp)
- Try FP32 instead of FP16

### Firestore Connection Issues
- Verify credentials file path
- Check internet connection
- Enable offline queue in config

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Adding New Features
1. Update `config.yaml` with new parameters
2. Modify relevant modules
3. Update documentation
4. Test thoroughly before deployment

