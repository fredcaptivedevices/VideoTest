# Video QA Automation for Hardware Validation

A robust QA automation application for validating dual-stream video recordings from head-mounted camera systems. This tool processes Camera A/B video streams and their associated metadata to validate recording integrity, dropped frame detection, and timecode synchronisation.

## Overview

This system implements a comprehensive validation pipeline that:

1. **Parses metadata** from JSON files containing frame counts, drop reports, and timecode information
2. **Analyses video frames** using computer vision to detect:
   - Visual timecode from digital slate displays (Ambient LockitSlate Take2 or similar)
   - "Dropped frame" indicator overlays (fixed ROI in top-left corner)
   - Dropped frames via timecode value comparison
3. **Applies Truth Table logic** to classify each frame and detect discrepancies
4. **Generates detailed reports** in multiple formats (HTML, Text, JSON, CSV)

## High Frame Rate Support

The system is designed to work with high frame rate recordings where the video frame rate differs from the timecode rate.

### Ambient LockitSlate Integration

The [Ambient LockitSlate Take2](https://ambient.de) is a professional timecode slate that displays SMPTE timecode via LED display. Key characteristics:

- Supports frame rates: 23.98, 24, 25, 29.97, 29.97DF, 30, 47.95, 48, 50, 59.94, 59.94DF, 60 FPS
- LED digits update at the timecode rate, not the video capture rate
- When shooting 60fps video with 30fps timecode, each timecode value appears on exactly 2 consecutive video frames

### Automatic Frame Rate Multiplier

The system **automatically computes** the frame rate multiplier from the metadata JSON:

| Video FPS (from metadata) | Timecode FPS | Multiplier | Expected Duplicates |
|---------------------------|--------------|------------|---------------------|
| 30 | 30 | 1 | 0 (each TC unique) |
| 60 | 30 | 2 | 1 (pairs share TC) |
| 120 | 30 | 4 | 3 (quads share TC) |

The multiplier is computed as: `video_fps // timecode_fps` (default timecode FPS is 30)

**Detection Logic:**
- At multiplier=2: Seeing the same timecode twice is NORMAL
- At multiplier=2: Seeing the same timecode 3+ times indicates a DROPPED FRAME

### Dual-Camera ROI Calibration

Each camera may have different framing, so the system calibrates **both cameras separately**:

1. GUI opens for Camera A (left) - draw box around timecode
2. Press ENTER to confirm
3. GUI opens for Camera B (right) - draw box around timecode
4. Press ENTER to confirm
5. Both ROIs saved to `roi_config.json`

## Installation

### Quick Start (macOS)

```bash
# 1. Create project folder
mkdir -p ~/CaptiveQA
cd ~/CaptiveQA

# 2. Copy all the .py files, requirements.txt, and validate.sh here

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate virtual environment (do this every time you open Terminal)
source venv/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Make shell script executable
chmod +x validate.sh
```

**Note:** EasyOCR will automatically download required models (~100MB) on first run. No external OCR engine installation required.

See `SETUP_MACOS.md` for detailed step-by-step instructions with troubleshooting.

## Usage

### Batch Validation (Recommended)

The easiest way to validate multiple takes:

```bash
cd ~/CaptiveQA
source venv/bin/activate
./validate.sh ./data
```

This will:
1. Discover all take folders under the data directory
2. Group takes by shot (parent folder)
3. Open the ROI calibration GUI for each camera in each shot
4. Process all takes with the calibrated ROIs
5. Generate reports in each take folder

### ROI Calibration

The system requires you to manually select the timecode display region for **both cameras** at the start of each shot:

1. **Camera A window** opens showing the first frame
2. **Draw a TIGHT box around ONLY the timecode digits** (e.g., `02:37:24:12`)
   - Do NOT include the entire slate - just the LED display area
   - Include a small margin (~10 pixels) around the digits
   - The ROI should be roughly 400-600 pixels wide and 60-100 pixels tall
3. Press **ENTER** to confirm
4. **Camera B window** opens
5. Draw a tight box around the timecode display  
6. Press **ENTER** to confirm
7. Press **ESC** at any point to skip the entire shot

**Important:** A tight ROI dramatically improves OCR accuracy. If the ROI is too large (capturing the whole slate), OCR will fail frequently.

The ROIs are saved to `roi_config.json` and used for all takes in that shot.

### Debugging OCR Issues

If OCR accuracy is poor, check the `debug_ocr/` folder which contains:
- `camA_frame0001_roi.png` - The actual region being sent to OCR
- `camA_frame0001_thresh200.png` - Thresholded image

If the ROI images show too much of the slate (not just the timecode), recalibrate with a tighter selection.

### Command Line (Single Take)

```bash
cd ~/CaptiveQA
source venv/bin/activate
python video_qa_validator.py \
    --video-a /path/to/camera_left.mov \
    --video-b /path/to/camera_right.mov \
    --metadata-a /path/to/camera_left.json \
    --metadata-b /path/to/camera_right.json \
    --output /path/to/reports \
    --debug
```

### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--video-a` | `-va` | Yes | Path to Camera A video file (.mov) |
| `--video-b` | `-vb` | Yes | Path to Camera B video file (.mov) |
| `--metadata-a` | `-ma` | Yes | Path to Camera A metadata JSON |
| `--metadata-b` | `-mb` | Yes | Path to Camera B metadata JSON |
| `--output` | `-o` | No | Output directory for reports (default: qa_reports) |
| `--verbose` | `-v` | No | Enable verbose logging |
| `--debug` | `-d` | No | Save screenshots of flagged frames |

### Python API

```python
from video_qa_validator import DualCameraValidator
from pathlib import Path

validator = DualCameraValidator(
    video_a=Path("camera_left.mov"),
    video_b=Path("camera_right.mov"),
    metadata_a=Path("camera_left.json"),
    metadata_b=Path("camera_right.json"),
    output_dir=Path("reports")
)

# Run validation
validator.run_full_validation()

# Generate reports
text_path, html_path = validator.generate_report()
```

## Duplicate Detection Method

### Primary: Timecode Value Comparison

The system detects dropped frames by comparing **actual timecode values** extracted via OCR:

```
Frame 100: TC = 02:37:29:15 (first occurrence)
Frame 101: TC = 02:37:29:15 (second occurrence - NORMAL at 60fps/30tc)
Frame 102: TC = 02:37:29:15 (third occurrence - DROPPED FRAME!)
Frame 103: TC = 02:37:29:16 (new timecode)
```

This approach is more reliable than image comparison because:
- It ignores irrelevant visual changes (lighting, motion blur, noise)
- It focuses on what actually matters: did the timecode advance?
- It correctly handles the expected duplicate pattern at high frame rates

### Fallback: Image Comparison

If OCR fails, the system falls back to MSE (Mean Squared Error) comparison of the timecode region. This is less reliable and should be avoided by ensuring good ROI calibration.

## Truth Table Logic

### Logic Gate 1: Continuity & Corruption

Analyses timecode progression between consecutive frames:

| Condition | Classification |
|-----------|----------------|
| Timecode jumps > 100 frames | **CORRUPTION** |
| Timecode skips + no duplicate | **UNDETECTED_SKIP** |
| Timecode increments normally | NORMAL |

### Logic Gate 2: Drop Frame Verification

Cross-references physical frame state with indicator presence:

| Physical Duplicate | Indicator Present | Classification |
|--------------------|-------------------|----------------|
| True | True | **DROPPED_DETECTED** (Success) |
| True | False | **DROPPED_UNDETECTED** (False Negative) |
| False | True | **FALSE_POSITIVE** |
| False | False | NORMAL |

## Configuration

### ROI Configuration File

The ROI calibration tool saves settings to `roi_config.json` with separate ROIs for each camera:

```json
{
    "roi_a": {
        "x": 465,
        "y": 1404,
        "width": 627,
        "height": 146
    },
    "roi_b": {
        "x": 480,
        "y": 1390,
        "width": 620,
        "height": 150
    },
    "version": "2.0",
    "note": "Separate ROI for each camera"
}
```

### Frame Rate Configuration

The frame rate multiplier is **automatically computed** from the metadata. The default timecode FPS (30) can be changed in the code:

```python
# In video_qa_validator.py
DEFAULT_TIMECODE_FPS = 30  # LockitSlate default

# The multiplier is computed automatically:
# multiplier = video_fps // timecode_fps
```

## Metadata JSON Format

Expected format for camera metadata files:

```json
{
    "Captive": {
        "FrameRate": "60",
        "DroppedFrames": "58",
        "NumFrames": "7718",
        "FullPathname": "C:/path/to/video.mov",
        "Filename": "video.mov",
        "ShotHierarchy": "pull3/Take_007",
        "CorruptedAt": "-1",
        "DropPercentage": "0",
        "Severity": "0"
    },
    "DropEvents": "",
    "Timecode": {
        "Hours": "2",
        "Minutes": "55",
        "Seconds": "23",
        "Frames": "0"
    }
}
```

## Output Reports

### HTML Report
Interactive report with styled tables and status badges showing:
- Camera information
- Sync status
- Corruption status
- Drop frame accuracy matrix
- Start timecode validation
- Indicator health summary
- Overall pass/fail result

### Text Report
Plain text version suitable for logging and automated parsing.

### JSON Report
Machine-readable format containing all validation results and statistics.

### Frame CSV
Detailed frame-by-frame analysis including:
- Frame number
- Visual timecode (OCR result)
- Drop indicator status
- Duplicate detection result
- Classification status
- Notes

## Debug Mode

When `--debug` is enabled or `DEBUG = True` in the code, the system saves annotated screenshots of any frame classified as:
- CORRUPTION
- DROPPED_UNDETECTED (False Negative)
- FALSE_POSITIVE
- UNDETECTED_SKIP

Screenshots are saved to `debug_frames/` with naming convention:
```
cam{A|B}_frame{NNNNNN}_{STATUS}.png
```

## Project Structure

```
video_qa_automation/
├── video_qa_validator.py   # Main validation engine
├── batch_validate.py       # Batch processing for multiple takes
├── calibrate_roi.py        # GUI tool for ROI selection
├── validate.sh             # Shell script runner
├── test_runner.py          # Unit tests and demo
├── config.json             # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── SETUP_MACOS.md          # macOS installation guide
├── qa_reports/             # Generated reports (output)
└── debug_frames/           # Flagged frame screenshots (debug)
```

### Class Hierarchy

- **Timecode**: SMPTE timecode representation and arithmetic
- **FrameAnalysis**: Per-frame analysis result container
- **CameraMetadata**: JSON metadata parser
- **FrameReader**: Video I/O, OCR, and duplicate detection
- **VideoAnalyser**: Single-camera analysis engine
- **DualCameraValidator**: Orchestrates dual-stream validation

## Dependencies

- `opencv-python>=4.8.0` - Video processing and image analysis
- `numpy>=1.24.0` - Numerical operations
- `Pillow>=10.0.0` - Image handling
- `easyocr>=1.7.0` - OCR engine (better for seven-segment displays than Tesseract)
- `pandas>=2.0.0` - Data analysis and CSV generation

## Troubleshooting

### OCR Failures
- EasyOCR models download automatically on first run
- Check timecode display is visible and well-lit in the ROI
- Verify ROI calibration covers the entire timecode display
- Try adjusting LED brightness on the LockitSlate

### False Duplicates
- Verify `FRAME_RATE_MULTIPLIER` matches your recording setup
- Check ROI is correctly positioned on the timecode digits
- Ensure OCR is successfully reading timecodes (check raw OCR output)

### Missing Drop Indicators
- Verify `ROI_DROP` coordinates match actual overlay position
- Check indicator brightness threshold in the code

### ROI Calibration Issues
- Ensure the first frame of each shot shows the slate clearly
- Draw the ROI box tightly around just the timecode digits
- Include a small margin (10px padding is added automatically)

## Licence

Proprietary - Captive Devices Ltd.

## Version History

- **1.2.0** - Timecode-based duplicate detection, high frame rate support
- **1.1.0** - ROI calibration GUI, batch processing
- **1.0.0** - Initial release with dual-camera validation support
