# Real-Time Defect Detection System for Jetson Nano

A real-time dual-camera product inspection system using YOLO object detection to identify defects (holes and dents) on products moving through a production line. Optimized for NVIDIA Jetson Nano.

## Features

- **Dual Camera System**: Simultaneously monitors products from top and side angles
- **Real-time Defect Detection**: Identifies holes and dents using YOLO model
- **Smart Confirmation**: Requires multiple consecutive frame detections to reduce false positives
- **Backend Integration**: Sends inspection results to a backend API
- **Live Statistics**: Real-time dashboard showing inspection metrics
- **Non-blocking Communication**: Async backend communication prevents processing delays

## Requirements

- Python 3.8+
- 2 USB cameras (Camera 0: Top view, Camera 1: Side view)
- NVIDIA Jetson Nano or compatible device
- YOLO model file (`.engine` format)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NeonicFlames/Real-Time-Defect-Detection-System-for-Jetson-Nano.git
   cd Real-Time-Defect-Detection-System-for-Jetson-Nano
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy ultralytics requests
   ```

3. **Setup your YOLO model:**
   - Place your trained YOLO model at `/app/model8.engine` or update the path in the code
   - Model should detect 6 classes:
     - Class 0: SIDE_DENT
     - Class 1: SIDE_HOLE
     - Class 2: SIDE_PRODUCT
     - Class 3: TOP_DENT
     - Class 4: TOP_HOLE
     - Class 5: TOP_PRODUCT

## Configuration

### Backend URL
Set your backend URL using an environment variable:

**Linux/Mac:**
```bash
export BACKEND_URL="https://your-backend-api.vercel.app"
```

**Windows PowerShell:**
```powershell
$env:BACKEND_URL="https://your-backend-api.vercel.app"
```

Or edit the default URL in `run.py`:
```python
BACKEND_URL = os.getenv('BACKEND_URL', 'https://example.com/api')
```

### Camera Settings
Edit the `Config` class in `run.py`:
```python
class Config:
    CAMERA_WIDTH = 320          # Frame width
    CAMERA_HEIGHT = 240         # Frame height
    TARGET_FPS = 5              # Target frame rate
    DISPLAY_SCALE = 3.0         # Display scaling
    
    SWITCH_DELAY = 1.0          # Wait time before switching cameras
    DETECTION_DELAY = 2.0       # Wait time after switch before detecting
    
    CONFIRMATION_FRAMES = 2     # Required consecutive detections
    CONFIRMATION_MIN_TIME = 0.1 # Minimum detection time (seconds)
```

## Usage

Run the inspection system:
```bash
python run.py
```

**Controls:**
- Press `q` to quit

## How It Works

1. **Camera 0 (Top)** waits for a product to enter the ROI (Region of Interest)
2. Once confirmed, product tracking begins and system switches to **Camera 1 (Side)**
3. Both cameras detect defects while product is in view
4. When product exits Camera 1's ROI, inspection completes
5. Results are sent to the backend and system returns to Camera 0

### Defect Counting Logic
- **Top camera holes**: Counted and sent to backend
- **Side camera holes**: Detected but not counted (to avoid duplicates)
- **Dents**: Detected from both cameras

## Backend API

The system sends POST requests to `/api/data` with this JSON format:

```json
{
  "type": "product",
  "payload": {
    "product_id": 123,
    "status": "DEFECTIVE",
    "processing_time": 5.42,
    "defects": ["hole", "hole"],
    "timestamp": "2026-02-06T12:34:56.789"
  }
}
```

## Display Windows

### Main Window (CountifyTech Inspector)
- Live camera feed with detection boxes
- ROI line indicator (Yellow: ready, Red: waiting)
- Camera status and confirmation progress

### Statistics Window
- Active camera indicator
- Backend connection status
- Total inspections
- OK vs DEFECTIVE counts
- Hole count statistics
- Recent inspection results

## Troubleshooting

**Cameras not detected:**
- Check USB connections
- Verify camera IDs (may need to adjust camera indices)
- Test cameras: `ls /dev/video*` (Linux)

**Model loading errors:**
- Ensure model path is correct
- Verify model format (.engine for TensorRT)
- Check CUDA/TensorRT installation

**Backend connection failed:**
- Verify backend URL is accessible
- Check network connection
- System will continue offline (backend optional)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

