# macOS Setup Guide for Video QA Automation

Step-by-step instructions to install and run the Video QA Automation tool on macOS.

## Prerequisites

- macOS 10.15 (Catalina) or later
- Administrator access (for Homebrew installations)
- Terminal application

---

## Step 1: Install Homebrew (if not already installed)

Homebrew is the package manager for macOS. Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen prompts. After installation, you may need to add Homebrew to your PATH. The installer will tell you the exact commands, but typically:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify installation:

```bash
brew --version
```

---

## Step 2: Install Python 3.11+

macOS comes with Python, but we recommend installing a fresh version:

```bash
brew install python@3.11
```

Verify installation:

```bash
python3 --version
```

You should see `Python 3.11.x` or higher.

---

## Step 3: Install Tesseract OCR

Tesseract is the OCR engine used to read timecodes from video frames:

```bash
brew install tesseract
```

Verify installation:

```bash
tesseract --version
```

You should see version 5.x.

---

## Step 4: Create Project Directory

Create a dedicated folder for the QA tool:

```bash
mkdir -p ~/CaptiveQA
cd ~/CaptiveQA
```

---

## Step 5: Download the Application Files

Copy all the application files into `~/CaptiveQA/`. You should have:

```
~/CaptiveQA/
├── video_qa_validator.py
├── test_runner.py
├── config.json
├── requirements.txt
└── README.md
```

If you're copying from a USB drive or download:

```bash
cp /path/to/downloaded/files/* ~/CaptiveQA/
```

---

## Step 6: Create a Virtual Environment (Recommended)

Virtual environments keep dependencies isolated:

```bash
cd ~/CaptiveQA
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` at the beginning.

**Note:** You'll need to run `source venv/bin/activate` each time you open a new Terminal window before using the tool.

---

## Step 7: Install Python Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- opencv-python (video processing)
- numpy (numerical operations)
- Pillow (image handling)
- pytesseract (OCR wrapper)
- imagehash (duplicate detection)
- pandas (data analysis)

---

## Step 8: Verify Installation

Run the test suite to confirm everything works:

```bash
python test_runner.py
```

You should see output ending with:

```
======================================================================
ALL TESTS COMPLETED SUCCESSFULLY
======================================================================
```

---

## Step 9: Organise Your Data

The tool expects your data in this structure:

```
~/CaptiveQA/
├── video_qa_validator.py
├── batch_validate.py
├── validate.sh
├── ...
└── data/
    ├── pull3/
    │   ├── Take_001/
    │   │   ├── pull3_Take_001_left.mov
    │   │   ├── pull3_Take_001_right.mov
    │   │   ├── pull3_Take_001_left.json
    │   │   └── pull3_Take_001_right.json
    │   ├── Take_002/
    │   │   └── ...
    │   └── Take_007/
    │       └── ...
    └── another_shoot/
        └── Take_001/
            └── ...
```

Create the data directory:

```bash
mkdir -p ~/CaptiveQA/data
```

Copy or symlink your shoot folders into `data/`.

---

## Step 10: Run Validation

### Option A: Process All Takes (Recommended)

Process every take found in the data directory:

```bash
cd ~/CaptiveQA
./validate.sh
```

Or specify a different data location:

```bash
./validate.sh /Volumes/Shoots/ProjectX
```

### Option B: Process a Single Take

```bash
./validate.sh --take ./data/pull3/Take_007
```

### Option C: List Discovered Takes (Dry Run)

See what would be processed without actually running validation:

```bash
./validate.sh --list
```

### Command Options

| Flag | Purpose |
|------|---------|
| `--take PATH` | Process only the specified take folder |
| `--list` | List all discovered takes without processing |
| `--debug` | Save screenshots of flagged frames |
| `--verbose` | Show detailed progress logging |

---

## Step 11: View Reports

### Per-Take Reports

Each take gets its own report folder:

```
data/pull3/Take_007/
├── pull3_Take_007_left.mov
├── pull3_Take_007_right.mov
├── pull3_Take_007_left.json
├── pull3_Take_007_right.json
└── qa_reports/
    ├── qa_report_20241221_143052.html
    ├── qa_report_20241221_143052.txt
    ├── qa_report_20241221_143052.json
    ├── frame_analysis_cam_a_20241221_143052.csv
    └── frame_analysis_cam_b_20241221_143052.csv
```

### Batch Summary Report

When processing multiple takes, a summary is generated:

```
data/qa_reports/
├── batch_summary_20241221_143052.html
├── batch_summary_20241221_143052.txt
└── batch_summary_20241221_143052.json
```

The HTML summary opens automatically on macOS, showing:
- Total takes processed
- Pass/fail counts
- Pass rate percentage
- Per-take breakdown with links to individual reports

---

## Quick Reference: Daily Workflow

```bash
# 1. Open Terminal and navigate to project
cd ~/CaptiveQA

# 2. Copy new shoot data (or symlink)
cp -r /Volumes/Card/shoot_name ./data/

# 3. Run batch validation
./validate.sh

# 4. Review the HTML summary that opens automatically
#    Click through to individual take reports as needed
```

### Single Take Quick Check

```bash
cd ~/CaptiveQA
./validate.sh --take ./data/pull3/Take_007
```

---

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"

Ensure Tesseract is installed and accessible:

```bash
which tesseract
```

If nothing is returned, reinstall:

```bash
brew reinstall tesseract
```

### "No module named 'cv2'"

The virtual environment isn't activated or opencv isn't installed:

```bash
source venv/bin/activate
pip install opencv-python
```

### "Permission denied" errors

Make sure you own the files:

```bash
sudo chown -R $(whoami) ~/CaptiveQA
```

### Poor OCR accuracy on timecodes

The timecode display might be too dark or at an unusual angle. Options:
1. Ensure good lighting on the slate during recording
2. Adjust thresholds in `config.json`:
   ```json
   "thresholds": {
       "drop_indicator_brightness_threshold": 150
   }
   ```

### Video files won't open

Check the codec. The tool expects MJPEG in .mov container. Verify with:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 yourfile.mov
```

Should return `mjpeg`.

---

## Updating the Tool

To update to a newer version:

```bash
cd ~/CaptiveQA
source venv/bin/activate

# Backup existing files
cp video_qa_validator.py video_qa_validator.py.bak

# Copy new files
cp /path/to/new/video_qa_validator.py .

# Update dependencies if requirements.txt changed
pip install -r requirements.txt --upgrade
```

---

## Uninstalling

To completely remove the tool:

```bash
# Remove project directory
rm -rf ~/CaptiveQA

# Optionally remove Tesseract
brew uninstall tesseract
```

---

## Support

For issues specific to the QA tool, check the debug output in:
- `video_qa.log` - Detailed processing log
- `debug_frames/` - Screenshots of flagged frames (when `--debug` is used)

For Captive Devices hardware issues, contact support@captivedevices.com
