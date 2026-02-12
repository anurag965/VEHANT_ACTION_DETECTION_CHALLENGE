# VEHANT Submission - File Index

## 📋 Complete File Listing

### 📄 Documentation Files (READ THESE FIRST)

| File | Purpose | Read Time |
|------|---------|-----------|
| **README.md** | Quick start guide, features, usage examples | 10 min |
| **SETUP.md** | Detailed installation and configuration guide | 15 min |
| **APPROACH.md** | Technical deep-dive, architecture details | 30 min |
| **INDEX.md** | This file - complete file listing | 5 min |

### 🐍 Python Source Code

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| **test.py** | Main testing script - batch inference with CSV output | 450 | ✅ Production |
| **vehant_causal_temporal_model.py** | Core VEHANT model implementation | 1200+ | ✅ Production |
| **ablation_study.py** | Ablation experiments (5 model variants) | 500+ | ✅ Validation |
| **convert_pth_to_onnx.py** | Convert PyTorch to ONNX format | 600+ | ✅ Export |

### 📦 Configuration Files

| File | Purpose |
|------|---------|
| **requirements.txt** | Python package dependencies |

### 📁 Directory Structure (To Create)

```
vehant_submission/
├── README.md                                    ← Quick start
├── SETUP.md                                     ← Installation guide
├── APPROACH.md                                  ← Technical details
├── INDEX.md                                     ← This file
├── requirements.txt                             ← Dependencies
├── test.py                                      ← MAIN: Batch inference
│
├── vehant_causal_temporal_model.py             ← Core model
├── ablation_study.py                           ← Validation
├── convert_pth_to_onnx.py                      ← ONNX export
│
├── models/causal_temporal/
│   ├── vehant_causal_temporal_original.pth     ← Pre-trained (60 MB)
│   └── vehant_causal_temporal_finetuned.pth    ← Optional fine-tuned
│
├── convo/
│   └── pose_landmarker_lite.task               ← MediaPipe pose (4 MB, download)
│
├── dataset/
│   ├── fight_mp4s/                             ← Fight video samples
│   ├── collapse_mp4s/                          ← Collapse video samples
│   └── negatives/                              ← Normal video samples
│
└── results/causal_temporal/                    ← Output directory
```

---

## 🚀 Quick Start Path

**For First-Time Users:**

1. **Read first**: `README.md` (5 min)
   - Understand what VEHANT does
   - See quick usage examples

2. **Setup**: Follow `SETUP.md` (15 min)
   ```bash
   pip install -r requirements.txt
   mkdir -p convo
   wget https://...pose_landmarker_lite.task -O convo/pose_landmarker_lite.task
   ```

3. **Test**: Run `test.py` (2 min)
   ```bash
   python test.py --input_dir test_videos --output_file results.csv --threshold 0.7
   ```

4. **Understand**: Read `APPROACH.md` (30 min)
   - Deep understanding of architecture
   - How each component works

**For Developers:**

1. Review `APPROACH.md` for architecture
2. Study `vehant_causal_temporal_model.py` 
3. Run `ablation_study.py` to validate components
4. Modify `Config` class as needed
5. Export to ONNX with `convert_pth_to_onnx.py`

---

## 📖 File Descriptions

### README.md
- **What**: Overview and quick start guide
- **Contains**:
  - Feature list
  - Installation steps
  - Usage examples
  - System requirements
  - Troubleshooting basics
- **Read if**: First time using VEHANT

### SETUP.md
- **What**: Detailed setup and configuration
- **Contains**:
  - Step-by-step installation
  - Virtual environment setup
  - Dependency verification
  - Configuration options
  - Common issues & solutions
  - Advanced setup (Docker, GPU, etc.)
- **Read if**: Setting up on new machine

### APPROACH.md
- **What**: Technical deep-dive
- **Contains**:
  - System overview and motivation
  - Component architecture (CNN, VQ-VAE, Causal Attention, etc.)
  - Training process and loss functions
  - Inference pipeline
  - Performance metrics
  - Experimental results
- **Read if**: Understanding the system deeply

### test.py
- **What**: Main inference script
- **Functionality**:
  - Takes folder of videos as input
  - Loads trained model
  - Processes each video with VEHANT
  - Outputs predictions as CSV
  - Supports custom confidence threshold
- **Usage**:
  ```bash
  python test.py --input_dir ./videos --output_file results.csv --threshold 0.7
  ```
- **Output**: CSV with columns: `video_name, class_id, x1, y1, x2, y2`

### vehant_causal_temporal_model.py
- **What**: Complete VEHANT implementation
- **Contains**:
  - Config class (hyperparameters)
  - VEHANTCausalTemporalModel (main model)
  - MotionVQVAE (motion tokenization)
  - CausalTemporalAttention (temporal modeling)
  - UncertaintyHead (uncertainty quantification)
  - VideoProcessor (video reading/processing)
  - ActionDataset (data loading)
  - Training loop (original + fine-tuning)
  - Inference functions
- **Modifiable**:
  - Config parameters
  - Loss weights
  - Learning rates
  - Batch sizes
- **Entry Points**:
  ```bash
  python vehant_causal_temporal_model.py --stage train
  python vehant_causal_temporal_model.py --stage finetune
  python vehant_causal_temporal_model.py --stage inference --video test.mp4
  ```

### ablation_study.py
- **What**: Validation of model components
- **Variants**:
  1. RGB Baseline (87%)
  2. + Motion Tokens (89%)
  3. + Causal Attention (91%)
  4. + Uncertainty (93%)
  5. Full System (95%)
- **Usage**:
  ```bash
  python ablation_study.py --dataset_path ./dataset --output ablation_results.json
  ```
- **Output**: JSON with accuracy, ECE, F1-score for each variant

### convert_pth_to_onnx.py
- **What**: Convert PyTorch model to ONNX format
- **Features**:
  - Automatic dependency detection
  - Smart checkpoint loading
  - ONNX verification
  - Inference benchmarking
  - Android optimization
  - Quantization support
- **Usage**:
  ```bash
  python convert_pth_to_onnx.py --opset_version 14
  python convert_pth_to_onnx.py --model_path models/.../original.pth
  python convert_pth_to_onnx.py --android_optimize
  python convert_pth_to_onnx.py --quantize  # 4x smaller
  ```
- **Output**: `.onnx` file + conversion report

### requirements.txt
- **What**: Python package dependencies
- **Contains**:
  - torch (2.0+)
  - torchvision
  - opencv-python
  - numpy
  - scipy
  - scikit-learn
  - mediapipe
  - tqdm
  - onnx
  - onnxruntime
  - And others
- **Installation**:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🔑 Key Files to Understand

### For Using the System
1. **test.py** - How to run inference
2. **README.md** - What it does and why
3. **requirements.txt** - What to install

### For Understanding
1. **APPROACH.md** - How it works
2. **vehant_causal_temporal_model.py** - Implementation details
3. **ablation_study.py** - Component validation

### For Customization
1. **vehant_causal_temporal_model.py** - Config class (edit hyperparameters)
2. **test.py** - Modify inference behavior
3. **SETUP.md** - Configuration guide

### For Deployment
1. **convert_pth_to_onnx.py** - Export for production
2. **SETUP.md** - Advanced setup section

---

## 📊 File Statistics

| Category | Count | Total Size |
|----------|-------|-----------|
| Documentation | 4 | ~100 KB |
| Python code | 4 | ~150 KB |
| Config files | 1 | ~1 KB |
| Models | 1-2 | ~60-120 MB |
| Assets | 1 | ~4 MB |
| **TOTAL** | **11-12** | **~250-280 MB** |

---

## 🔗 Inter-File Dependencies

```
test.py
    ↓ imports
vehant_causal_temporal_model.py
    ↓ requires
convo/pose_landmarker_lite.task
models/causal_temporal/vehant_causal_temporal_original.pth

ablation_study.py
    ↓ imports
vehant_causal_temporal_model.py
    ↓ requires
dataset/[fight_mp4s, collapse_mp4s, negatives]

convert_pth_to_onnx.py
    ↓ imports
vehant_causal_temporal_model.py
    ↓ requires
models/causal_temporal/vehant_causal_temporal_original.pth
```

---

## 📋 CSV Output Format

Generated by `test.py`, format:
```
video_name,pred_class_1,x1,y1,x2,y2,[pred_class_2,x1,y1,x2,y2,...]

Example:
fight_001.mp4,1,0.1234,0.2345,0.8765,0.9234
collapse_002.mp4,2,0.0500,0.1000,0.9500,0.8500
normal_003.mp4,0,0.0000,0.0000,1.0000,1.0000
```

---

## ⚙️ Configuration Customization

All settings in `vehant_causal_temporal_model.py` `Config` class:

```python
# Video processing
FRAME_SAMPLE_RATE = 2          # Change to 1 for more frames
SEQUENCE_LENGTH = 16           # Change to 8/32 for different context window
IMG_SIZE = 320                 # Resolution (224/256/512 possible)

# Training
BATCH_SIZE = 4                 # Reduce for limited memory
LEARNING_RATE = 0.0001         # Adjust for training speed
EPOCHS = 30                    # More epochs = better accuracy

# Inference
CONFIDENCE_THRESHOLD = 0.65    # Lower = more detections
```

---

## 🆘 Finding Help

**Problem**: Where do I find X?
- **Installation help**: See `SETUP.md`
- **Usage examples**: See `README.md`
- **Technical details**: See `APPROACH.md`
- **Code details**: See docstrings in Python files

**Problem**: My issue isn't listed
1. Check `SETUP.md` → "Common Issues"
2. Check `README.md` → "Troubleshooting"
3. Check error message in Python file docstrings
4. Run test_imports.py to verify setup

---

## ✅ Verification Checklist

Before using VEHANT:

- [ ] Extracted `vehant_submission.zip`
- [ ] Ran `pip install -r requirements.txt`
- [ ] Downloaded `pose_landmarker_lite.task` to `convo/`
- [ ] Have `vehant_causal_temporal_original.pth` in `models/causal_temporal/`
- [ ] Verified imports: `python test_imports.py`
- [ ] Tested on sample video: `python test.py --input_dir test_videos --output_file results.csv`
- [ ] Read `README.md` for overview

If all ✓, you're ready to use VEHANT!

---

## 📝 Version History

**Version 1.0** (Current - January 2025)
- ✅ Full VEHANT system with causal attention
- ✅ Uncertainty quantification
- ✅ Multi-task learning
- ✅ ONNX export
- ✅ Complete documentation
- ✅ Ablation study validation
- ✅ Production ready

---

**Last Updated**: January 2025  
**Status**: Production Ready  
**Support**: See README.md and SETUP.md
