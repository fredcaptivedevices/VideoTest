# Windows Setup Guide for Video QA Automation (with GPU Acceleration)

Complete step-by-step instructions to install and run the Video QA Automation tool on Windows with NVIDIA GPU acceleration.

## Prerequisites

- Windows 10 or 11
- NVIDIA GPU (GTX 1060 or better recommended)
- Administrator access
- ~2GB free disk space

---

## Step 1: Install NVIDIA GPU Drivers

Make sure you have the latest NVIDIA drivers installed.

1. Go to https://www.nvidia.com/Download/index.aspx
2. Select your GPU model
3. Download and install the latest driver
4. Restart your computer

Verify installation by opening Command Prompt and running:

```cmd
nvidia-smi
```

You should see your GPU listed with driver version.

---

## Step 2: Install Python 3.11

1. Download Python from https://www.python.org/downloads/
2. **Important:** During installation, check the box **"Add Python to PATH"**
3. Click "Install Now"

Verify installation by opening a **new** Command Prompt:

```cmd
python --version
```

You should see `Python 3.11.x`.

---

## Step 3: Create Project Folder

Open Command Prompt and run:

```cmd
mkdir C:\CaptiveQA
cd C:\CaptiveQA
```

---

## Step 4: Copy Application Files

Copy all the application files into `C:\CaptiveQA\`. You need:

```
C:\CaptiveQA\
├── video_qa_validator.py
├── batch_validate.py
├── calibrate_roi.py
├── requirements.txt
├── README.md
└── SETUP_WINDOWS.md
```

---

## Step 5: Create Virtual Environment

```cmd
cd C:\CaptiveQA
python -m venv venv
```

---

## Step 6: Activate Virtual Environment

**You must do this every time you open a new Command Prompt:**

```cmd
cd C:\CaptiveQA
venv\Scripts\activate
```

Your prompt will change to show `(venv)` at the beginning:

```
(venv) C:\CaptiveQA>
```

---

## Step 7: Upgrade pip

```cmd
python -m pip install --upgrade pip
```

---

## Step 8: Install PyTorch with CUDA (GPU Support)

This is the key step for GPU acceleration. Install PyTorch with CUDA 12.1 support:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

This download is ~2.5GB - be patient.

**Alternative CUDA versions:**
- For CUDA 11.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

---

## Step 9: Install Other Dependencies

```cmd
pip install opencv-python numpy pandas easyocr
```

---

## Step 10: Verify GPU Detection

Test that PyTorch can see your GPU:

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see:

```
CUDA available: True
GPU: NVIDIA GeForce RTX 3080  (or your GPU name)
```

If it says `CUDA available: False`, see Troubleshooting below.

---

## Step 11: Verify Full Installation

```cmd
python -c "import cv2; import easyocr; import pandas; import torch; print(f'All OK - GPU: {torch.cuda.is_available()}')"
```

---

## Step 12: Organise Your Data

Create a data folder and copy your video files:

```cmd
mkdir C:\CaptiveQA\data
```

Structure should be:

```
C:\CaptiveQA\data\
└── pull3\
    ├── Take_001\
    │   ├── pull3_Take_001_left.mov
    │   ├── pull3_Take_001_right.mov
    │   ├── pull3_Take_001_left.json
    │   └── pull3_Take_001_right.json
    └── Take_002\
        └── ...
```

---

## Step 13: Run Validation

```cmd
cd C:\CaptiveQA
venv\Scripts\activate
python batch_validate.py data
```

### What Happens

1. The tool discovers all take folders
2. For each shot, GUI windows open to select timecode ROI:
   - **Camera A** window - draw box around timecode, press ENTER
   - **Camera B** window - draw box around timecode, press ENTER
3. All takes are processed using your GPU
4. Reports are generated in each take folder

You should see a message like:

```
[OCR] Using GPU: NVIDIA GeForce RTX 3080
```

This confirms GPU acceleration is active.

---

## Daily Workflow

```cmd
:: 1. Open Command Prompt

:: 2. Navigate and activate
cd C:\CaptiveQA
venv\Scripts\activate

:: 3. Run validation
python batch_validate.py data

:: 4. View reports in data\<shot>\<take>\qa_reports\
```

---

## Command Options

| Command | Purpose |
|---------|---------|
| `python batch_validate.py data` | Process all takes |
| `python batch_validate.py --take data\pull3\Take_007` | Process single take |
| `python batch_validate.py --list data` | List discovered takes |
| `python batch_validate.py --debug data` | Save debug screenshots |

---

## Troubleshooting

### "CUDA available: False" but I have an NVIDIA GPU

1. **Check NVIDIA driver version:**
   ```cmd
   nvidia-smi
   ```
   Driver should be 450.x or newer.

2. **Reinstall PyTorch with correct CUDA version:**
   
   Check your CUDA version in nvidia-smi output (top right), then install matching PyTorch:
   
   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Try CUDA 11.8 if 12.1 doesn't work:**
   ```cmd
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### "python is not recognized"

Python wasn't added to PATH. Either:
- Reinstall Python and check "Add to PATH"
- Or use the full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe`

### "No module named 'cv2'"

Activate the virtual environment first:

```cmd
cd C:\CaptiveQA
venv\Scripts\activate
pip install opencv-python
```

### GUI windows don't appear

Make sure you're running on a machine with a display, not via remote desktop (which can have issues with OpenCV windows).

### Out of GPU memory

If you get CUDA out of memory errors, your GPU may not have enough VRAM. The tool will still work but may be slower. EasyOCR typically needs 2-4GB VRAM.

---

## Performance Expectations

With GPU acceleration:
- **With GPU (RTX 3080):** ~20-50 frames/second
- **Without GPU (CPU only):** ~2-5 frames/second

A 7000 frame take should process in about 2-5 minutes with a good GPU.

---

## Updating the Tool

```cmd
cd C:\CaptiveQA
venv\Scripts\activate

:: Backup old files
mkdir backups
copy *.py backups\

:: Copy new files over
:: (copy new .py files to C:\CaptiveQA)

:: Update dependencies if needed
pip install -r requirements.txt --upgrade
```

---

## Complete Reinstallation

If things are broken:

```cmd
cd C:\CaptiveQA

:: Remove old venv
rmdir /s /q venv

:: Create fresh environment
python -m venv venv
venv\Scripts\activate

:: Install PyTorch with GPU first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: Install other packages
pip install opencv-python numpy pandas easyocr
```

---

## Support

For tool issues, check:
- `video_qa.log` - processing log
- `debug_frames\` - screenshots of flagged frames

For Captive Devices hardware issues: support@captivedevices.com
