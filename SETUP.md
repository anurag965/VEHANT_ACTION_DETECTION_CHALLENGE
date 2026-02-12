# VEHANT Setup Guide

Complete step-by-step instructions for setting up and running VEHANT.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Common Issues](#common-issues)
6. [Advanced Setup](#advanced-setup)

---

## Prerequisites

### System Requirements

**Minimum**:
- Python 3.8 or higher
- 4 GB RAM
- 10 GB disk space
- CPU with SSE4.2 support

**Recommended**:
- Python 3.9 or 3.10
- 16 GB RAM
- 20 GB disk space (for models and data)
- NVIDIA GPU with CUDA 11.8+

### Supported Platforms
- **Linux**: Ubuntu 20.04+, Debian 10+, CentOS 8+
- **macOS**: Intel Mac (10.13+) or Apple Silicon (M1+)
- **Windows**: Windows 10 Version 2004+, Windows 11

### Check Your Python Version

```bash
python --version
# Expected output: Python 3.8.x or higher
```

If you have multiple Python versions, use `python3`:

```bash
python3 --version
```

---

## Installation

### Step 1: Extract the Archive

```bash
# Download the submission file
unzip vehant_submission.zip

# Navigate to the directory
cd vehant_submission

# Verify contents
ls -la
# Should see: test.py, README.md, APPROACH.md, requirements.txt, etc.
```

### Step 2: Create Virtual Environment

A virtual environment isolates project dependencies.

**Linux/macOS**:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

**Windows (Command Prompt)**:
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip, setuptools, wheel

```bash
# Important: upgrade pip first to avoid installation issues
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Packages

```bash
# Install from requirements.txt
pip install -r requirements.txt

# This installs:
# - torch (PyTorch)
# - torchvision
# - opencv-python
# - numpy
# - scikit-learn
# - mediapipe
# - tqdm
# - onnx/onnxruntime
# - And others
```

**Installation Time**: 10-30 minutes (depending on internet speed)

**Verify Installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import mediapipe; print('MediaPipe: OK')"
```

### Step 5: Download MediaPipe Pose Model

The pose detection model must be downloaded separately.

**Linux/macOS**:
```bash
# Create directory
mkdir -p convo

# Download pose model (required for skeleton features)
cd convo
wget https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task
cd ..

# Verify download
ls -lh convo/pose_landmarker_lite.task
# Should show ~4 MB file
```

**Windows (using PowerShell)**:
```powershell
# Create directory
mkdir convo

# Download using curl
curl -o convo\pose_landmarker_lite.task `
     "https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task"

# Verify
dir convo
```

**Alternative: Manual Download**
```
1. Go to: https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task
2. Right-click → Save as
3. Save to: vehant_submission/convo/pose_landmarker_lite.task
```

### Step 6: Verify Installation

```bash
# Check all imports work
python test_imports.py
```

Create `test_imports.py`:
```python
#!/usr/bin/env python3
"""Verify all dependencies are installed correctly."""

import sys

print("Checking imports...")
errors = []

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    errors.append(f"✗ PyTorch: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    errors.append(f"✗ OpenCV: {e}")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except ImportError as e:
    errors.append(f"✗ NumPy: {e}")

try:
    import mediapipe
    print(f"✓ MediaPipe installed")
except ImportError as e:
    errors.append(f"✗ MediaPipe: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    errors.append(f"✗ scikit-learn: {e}")

try:
    from vehant_causal_temporal_model import VEHANTCausalTemporalModel
    print(f"✓ VEHANT model imports successfully")
except ImportError as e:
    errors.append(f"✗ VEHANT model: {e}")

# Check GPU
print("\nGPU Information:")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  NVIDIA drivers: OK")
else:
    print("ℹ CUDA not available, will use CPU (slower)")

# Check pose model
import os
pose_model_path = "convo/pose_landmarker_lite.task"
if os.path.exists(pose_model_path):
    size_mb = os.path.getsize(pose_model_path) / (1024*1024)
    print(f"✓ Pose model found ({size_mb:.1f} MB)")
else:
    errors.append(f"✗ Pose model not found at {pose_model_path}")
    print("  Run: mkdir -p convo && cd convo && wget ...")

if errors:
    print("\n❌ Setup incomplete:")
    for error in errors:
        print(f"  {error}")
    sys.exit(1)
else:
    print("\n✅ All checks passed! Ready to use VEHANT.")
    sys.exit(0)
```

Run it:
```bash
python test_imports.py
```

---

## Configuration

### Default Configuration

All settings are in `vehant_causal_temporal_model.py`:

```python
class Config:
    # Paths
    DATASET_PATH = 'dataset'
    CUSTOM_DATA_DIR = 'custom_data'
    MODEL_SAVE_PATH = 'models/causal_temporal'
    RESULTS_PATH = 'results/causal_temporal'
    
    # Video Processing
    FRAME_SAMPLE_RATE = 2              # Sample every 2nd frame
    SEQUENCE_LENGTH = 16               # Process 16 frames
    IMG_SIZE = 320                     # Resize frames to 320×320
    OPTICAL_FLOW_SIZE = 64             # Optical flow resolution
    
    # Training
    BATCH_SIZE = 4                     # Batch size for training
    GRAD_ACCUMULATION_STEPS = 2        # Gradient accumulation
    EPOCHS = 30                        # Number of training epochs
    LEARNING_RATE = 0.0001            # Initial learning rate
    PATIENCE = 10                      # Early stopping patience
    WARMUP_EPOCHS = 5                  # Warmup epochs
    
    # Fine-tuning
    FINETUNE_EPOCHS = 40
    FINETUNE_LR = 0.0001
    FINETUNE_PATIENCE = 8
    
    # Model Architecture
    NUM_CLASSES = 3                    # Action classes
    MOTION_VOCAB_SIZE = 256            # VQ-VAE codebook size
    HIDDEN_DIM = 448                   # Feature dimension
    NUM_HEADS = 8                      # Attention heads
    NUM_LAYERS = 2                     # Transformer layers
    
    # Classes
    CLASS_NAMES = ['negative', 'fight', 'collapse']
    
    # Inference
    CONFIDENCE_THRESHOLD = 0.65        # Detection threshold
    NEGATIVE_CLASS_MIN_PROB = 0.25     # Soft rejection threshold
```

### Adjusting for Your Hardware

**For GPU with Limited Memory**:
```python
BATCH_SIZE = 2          # Reduce from 4 to 2
GRAD_ACCUMULATION_STEPS = 4  # Increase to maintain gradient scale
```

**For CPU-only (slow)**:
```python
BATCH_SIZE = 1
SEQUENCE_LENGTH = 8     # Use shorter sequences
```

**For Fast Inference**:
```python
CONFIDENCE_THRESHOLD = 0.5  # Lower threshold = more detections
```

**For High Precision** (fewer false positives):
```python
CONFIDENCE_THRESHOLD = 0.8  # Higher threshold = fewer detections
```

### Create Directory Structure

```bash
# Create required directories
mkdir -p models/causal_temporal
mkdir -p results/causal_temporal
mkdir -p dataset/{fight_mp4s,collapse_mp4s,negatives}
mkdir -p custom_data/{fight,collapse,negative}
mkdir -p convo
```

---

## Verification

### Verify the Installation

#### 1. Check Python Version
```bash
python --version
# Should output: Python 3.8.x or higher
```

#### 2. Check GPU (if applicable)
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

#### 3. Check Model File
```bash
ls -lh models/causal_temporal/vehant_causal_temporal_original.pth
# Should show a ~60 MB file
```

#### 4. Check Pose Model
```bash
ls -lh convo/pose_landmarker_lite.task
# Should show a ~4 MB file
```

#### 5. Test Import
```bash
python -c "from vehant_causal_temporal_model import VEHANTCausalTemporalModel; print('✓ Model imports OK')"
```

### First Run Test

Create a simple test video first:

```bash
# Create a dummy test video (30 frames, 320×320)
python -c "
import cv2
import numpy as np

out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 320))
for i in range(30):
    frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    out.write(frame)
out.release()
print('Created test.mp4')
"

# Create test directory
mkdir test_videos
mv test.mp4 test_videos/

# Run test
python test.py --input_dir test_videos --output_file test_results.csv

# Check output
cat test_results.csv
```

Expected output:
```
test.mp4,0,0.0000,0.0000,1.0000,1.0000
```

---

## Common Issues

### Issue 1: ModuleNotFoundError: No module named 'mediapipe'

**Error Message**:
```
ModuleNotFoundError: No module named 'mediapipe'
```

**Solution**:
```bash
# Reinstall mediapipe
pip install --upgrade mediapipe

# If still fails, install dependencies
pip install google-protobuf numpy opencv-python
pip install mediapipe
```

### Issue 2: Pose Model Not Found

**Error Message**:
```
āœ CRITICAL: Pose model not found at convo/pose_landmarker_lite.task
```

**Solution**:
```bash
# Create directory
mkdir -p convo

# Download model (Linux/macOS)
cd convo
wget https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task
cd ..

# Or download manually from:
# https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task
# Save to: convo/pose_landmarker_lite.task
```

### Issue 3: CUDA Out of Memory

**Error Message**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solution**:
```python
# Edit vehant_causal_temporal_model.py
class Config:
    BATCH_SIZE = 1          # Reduce from 4 to 1
    GRAD_ACCUMULATION_STEPS = 8  # Increase to compensate
    SEQUENCE_LENGTH = 8     # Optional: use shorter sequences
```

Or use CPU:
```bash
# Force CPU
CUDA_VISIBLE_DEVICES="" python test.py --input_dir videos --output_file results.csv
```

### Issue 4: No Video Files Found

**Error Message**:
```
✗ Error: No video files found in ./videos
  Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm
```

**Solution**:
```bash
# Check directory exists
ls -la videos/

# Check video format
file videos/*.mp4
# Should show: "ISO Media, MP4 Base Media v1 [IS0 14496-12:2005]"

# Check video is readable
python -c "
import cv2
cap = cv2.VideoCapture('videos/test.mp4')
if cap.isOpened():
    print('✓ Video readable')
else:
    print('✗ Video not readable')
"
```

### Issue 5: Slow Inference on CPU

**Expected Speed**:
- GPU: 25 ms
- CPU: 200-300 ms

**Optimization Tips**:
```bash
# 1. Use GPU if available
python test.py --input_dir videos --output_file results.csv

# 2. Reduce sequence length
# Edit Config: SEQUENCE_LENGTH = 8

# 3. Use ONNX quantized model (4x faster)
python convert_pth_to_onnx.py --quantize

# 4. Reduce frame resolution
# Edit Config: IMG_SIZE = 224
```

### Issue 6: Python Environment Issues

**Symptom**: Commands work sometimes but not others

**Solution**:
```bash
# Always activate virtual environment first
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Verify activation (should see (venv) in prompt)
which python
# Should show: /path/to/vehant_submission/venv/bin/python
```

### Issue 7: Import Errors on macOS with Apple Silicon

**Symptom**: ImportError with some packages

**Solution**:
```bash
# Use conda instead of venv
conda create -n vehant python=3.10
conda activate vehant

# Install PyTorch for Apple Silicon
conda install pytorch::pytorch torchvision -c pytorch

# Install other packages
pip install -r requirements.txt
```

---

## Advanced Setup

### GPU Setup (NVIDIA)

**1. Check NVIDIA Driver**:
```bash
nvidia-smi
# Should show: NVIDIA GPU, CUDA Compute Capability 7.0+
```

**2. Install CUDA Toolkit** (if not installed):
```bash
# Visit: https://developer.nvidia.com/cuda-downloads
# Download CUDA 11.8 or 12.1
# Follow installation instructions
```

**3. Install cuDNN** (for faster operations):
```bash
# Visit: https://developer.nvidia.com/cudnn
# Download and install cuDNN 8.x
```

**4. Verify CUDA in PyTorch**:
```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
"
```

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

WORKDIR /workspace

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Download pose model
RUN mkdir -p convo && \
    wget https://storage.googleapis.com/mediapipe-tasks/python/pose_landmarker_lite.task \
         -O convo/pose_landmarker_lite.task

ENTRYPOINT ["python", "test.py"]
```

Build and run:
```bash
# Build
docker build -t vehant:latest .

# Run
docker run --gpus all -v /data:/workspace/data vehant:latest \
    --input_dir /data/videos \
    --output_file /data/results.csv
```

### Using Different Python Versions

```bash
# List available Python versions
ls /usr/bin/python*

# Create venv with specific version
python3.9 -m venv venv_py39
source venv_py39/bin/activate

# Install packages
pip install -r requirements.txt
```

### Conda Environment (Alternative)

```bash
# Create environment
conda create -n vehant python=3.10

# Activate
conda activate vehant

# Install PyTorch (conda has better binary compatibility)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install -r requirements.txt
```

---

## Next Steps

After successful setup:

1. **Run basic test**: `python test.py --input_dir test_videos --output_file results.csv`
2. **Read README.md**: Overview of VEHANT features
3. **Read APPROACH.md**: Technical details and architecture
4. **Prepare your data**: Organize videos by class
5. **Run inference**: Process your videos
6. **Analyze results**: Check CSV output

---

## Support Commands

### Quick Health Check
```bash
bash -c "
echo 'Python:' && python --version && \
echo 'PyTorch:' && python -c 'import torch; print(torch.__version__)' && \
echo 'GPU:' && python -c 'import torch; print(\"Available\" if torch.cuda.is_available() else \"Not available\")' && \
echo 'Pose model:' && ls -lh convo/pose_landmarker_lite.task 2>/dev/null && echo 'OK' || echo 'Missing'
"
```

### View Installed Packages
```bash
pip list | grep -E "(torch|opencv|mediapipe|onnx)"
```

### Clear Cache and Reinstall
```bash
# Remove cache
rm -rf ~/.cache/pip
pip cache purge

# Reinstall
pip install --no-cache-dir -r requirements.txt
```

---

**Setup Complete!** You're ready to use VEHANT.
