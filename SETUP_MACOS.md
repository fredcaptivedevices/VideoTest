# macOS Setup Guide for Video QA Automation

Complete step-by-step instructions to install and run the Video QA Automation tool on macOS.

## Prerequisites

- macOS 10.15 (Catalina) or later
- Administrator access (for Homebrew installations)
- Terminal application (in Applications > Utilities)
- ~500MB free disk space (for Python, dependencies, and OCR models)

---

## Step 1: Install Homebrew (Package Manager)

Homebrew is required to install Python. Open **Terminal** and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen prompts (you'll need to enter your password).

**Important:** After installation, Homebrew will display commands to add it to your PATH. Run those commands. They typically look like:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify Homebrew is installed:

```bash
brew --version
```

You should see something like `Homebrew 4.x.x`.

---

## Step 2: Install Python 3.11

macOS comes with an old Python. Install a current version:

```bash
brew install python@3.11
```

Verify installation:

```bash
python3 --version
```

You should see `Python 3.11.x` or higher.

**Note:** On macOS, always use `python3` and `pip3` (not `pip` or `python`).

---

## Step 3: Create Project Directory

Create a folder for the QA tool:

```bash
mkdir -p ~/CaptiveQA
cd ~/CaptiveQA
```

---

## Step 4: Copy Application Files

Copy all the application files you received into `~/CaptiveQA/`. You need:

```
~/CaptiveQA/
├── video_qa_validator.py    (main validation engine)
├── batch_validate.py        (batch processing)
├── calibrate_roi.py         (ROI selection GUI)
├── validate.sh              (shell script runner)
├── requirements.txt         (Python dependencies)
├── README.md                (documentation)
└── SETUP_MACOS.md           (this file)
```

You can copy from Finder, or from Terminal:

```bash
# Example: copying from Downloads
cp ~/Downloads/video_qa_files/* ~/CaptiveQA/
```

---

## Step 5: Create a Virtual Environment

A virtual environment keeps dependencies isolated from your system Python.

```bash
cd ~/CaptiveQA
python3 -m venv venv
```

This creates a `venv` folder containing a private Python installation.

---

## Step 6: Activate the Virtual Environment

**You must do this every time you open a new Terminal window before using the tool.**

```bash
cd ~/CaptiveQA
source venv/bin/activate
```

Your terminal prompt will change to show `(venv)` at the beginning:

```
(venv) yourname@MacBook CaptiveQA %
```

---

## Step 7: Upgrade pip

Before installing dependencies, upgrade pip:

```bash
pip install --upgrade pip
```

**Note:** Inside the virtual environment, you can use `pip` instead of `pip3`.

---

## Step 8: Install Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- **opencv-python** - video processing
- **numpy** - numerical operations
- **pandas** - data analysis and CSV generation
- **easyocr** - OCR engine for reading timecodes

**First-time note:** EasyOCR will download recognition models (~100MB) on first run. This is automatic and only happens once.

---

## Step 9: Make the Shell Script Executable

```bash
chmod +x validate.sh
```

---

## Step 10: Verify Installation

Test that everything works:

```bash
python -c "import cv2; import easyocr; import pandas; print('All dependencies OK')"
```

You should see `All dependencies OK`.

---

## Step 11: Organise Your Data

The tool expects your video data in this structure:

```
~/CaptiveQA/
├── video_qa_validator.py
├── batch_validate.py
├── ...
└── data/
    └── pull3/                    (shot folder)
        ├── Take_001/             (take folder)
        │   ├── pull3_Take_001_left.mov
        │   ├── pull3_Take_001_right.mov
        │   ├── pull3_Take_001_left.json
        │   └── pull3_Take_001_right.json
        ├── Take_002/
        │   └── ...
        └── Take_007/
            └── ...
```

Create the data directory:

```bash
mkdir -p ~/CaptiveQA/data
```

Copy or move your shoot folders into `data/`:

```bash
# Example: copy from external drive
cp -r /Volumes/SSD/pull3 ~/CaptiveQA/data/
```

---

## Step 12: Run Validation

### Quick Start

```bash
cd ~/CaptiveQA
source venv/bin/activate
./validate.sh ./data
```

### What Happens

1. The tool discovers all take folders
2. For each shot, it opens a GUI to select the timecode ROI:
   - **Camera A** window appears first - draw a box around the timecode display
   - Press **ENTER** to confirm
   - **Camera B** window appears - draw a box around the timecode display
   - Press **ENTER** to confirm
3. All takes in that shot are processed
4. Reports are generated in each take folder

### Command Options

| Command | Purpose |
|---------|---------|
| `./validate.sh ./data` | Process all takes in data folder |
| `./validate.sh --take ./data/pull3/Take_007` | Process single take |
| `./validate.sh --list ./data` | List discovered takes (dry run) |
| `./validate.sh --debug ./data` | Save debug screenshots |

---

## Step 13: View Reports

### Per-Take Reports

Each take gets a `qa_reports` subfolder:

```
data/pull3/Take_007/
├── pull3_Take_007_left.mov
├── pull3_Take_007_right.mov
├── ...
└── qa_reports/
    ├── qa_report_20241221_143052.html   ← Open this in browser
    ├── qa_report_20241221_143052.txt
    ├── qa_report_20241221_143052.json
    ├── frame_analysis_cam_a_*.csv
    └── frame_analysis_cam_b_*.csv
```

Open the HTML report in your browser for a formatted view.

### Batch Summary

When processing multiple takes, a summary is generated in the data folder.

---

## Daily Workflow (Quick Reference)

```bash
# 1. Open Terminal

# 2. Navigate to project and activate environment
cd ~/CaptiveQA
source venv/bin/activate

# 3. Copy new data (if needed)
cp -r /Volumes/Card/shoot_name ./data/

# 4. Run validation
./validate.sh ./data

# 5. Review HTML reports in each take's qa_reports folder
```

---

## Troubleshooting

### "command not found: pip" or "pip: command not found"

You need to activate the virtual environment first:

```bash
cd ~/CaptiveQA
source venv/bin/activate
```

Or use `pip3` instead of `pip` if not in a virtual environment.

### "No module named 'cv2'" or "No module named 'easyocr'"

Dependencies aren't installed. Activate venv and install:

```bash
cd ~/CaptiveQA
source venv/bin/activate
pip install -r requirements.txt
```

### "zsh: permission denied: ./validate.sh"

Make the script executable:

```bash
chmod +x validate.sh
```

### EasyOCR downloads models every time

Models should be cached in `~/.EasyOCR/`. If they keep downloading, check disk space and permissions:

```bash
ls -la ~/.EasyOCR/
```

### OCR accuracy is poor

1. Make sure ROI selection covers the entire timecode display with some padding
2. Ensure the slate LED display is bright and visible in the video
3. Check that lighting doesn't cause glare on the LED display

### "Cannot connect to X server" or GUI doesn't appear

The ROI calibration requires a display. If running via SSH, you need X11 forwarding:

```bash
ssh -X user@host
```

Or run directly on the Mac with a display connected.

### Video files won't open

Check the codec:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 yourfile.mov
```

---

## Updating the Tool

When you receive updated files:

```bash
cd ~/CaptiveQA
source venv/bin/activate

# Backup existing files (optional)
mkdir -p backups
cp *.py backups/

# Copy new files
cp /path/to/new/files/*.py .
cp /path/to/new/files/requirements.txt .

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## Complete Reinstallation

If things are broken, start fresh:

```bash
# Remove old installation
rm -rf ~/CaptiveQA/venv

# Create new virtual environment
cd ~/CaptiveQA
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## File Reference

| File | Purpose |
|------|---------|
| `video_qa_validator.py` | Main validation engine |
| `batch_validate.py` | Batch processing for multiple takes |
| `calibrate_roi.py` | GUI for selecting timecode region |
| `validate.sh` | Shell script to run validation |
| `requirements.txt` | Python package dependencies |
| `README.md` | Full documentation |
| `SETUP_MACOS.md` | This setup guide |

---

## Support

For tool issues, check:
- `video_qa.log` - processing log
- `debug_frames/` - screenshots of flagged frames (with `--debug`)

For Captive Devices hardware issues: support@captivedevices.com
