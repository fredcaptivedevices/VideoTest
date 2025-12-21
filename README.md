# Video QA Automation for Hardware Validation

A robust QA automation application for validating dual-stream video recordings from head-mounted camera systems. This tool processes Camera A/B video streams and their associated metadata to validate recording integrity, dropped frame detection, and timecode synchronisation.

## Overview

This system implements a comprehensive validation pipeline that:

1. **Parses metadata** from JSON files containing frame counts, drop reports, and timecode information
2. **Analyses video frames** using computer vision to detect:
   - Visual timecode from digital slate displays (auto-detected ROI)
   - "Dropped frame" indicator overlays (fixed ROI in top-left corner)
   - Physical frame duplicates (freeze frames indicating actual drops)
3. **Applies Truth Table logic** to classify each frame and detect discrepancies
4. **Generates detailed reports** in multiple formats (HTML, Text, JSON, CSV)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Tesseract OCR is installed
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Command Line

```bash
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

## Truth Table Logic

The system implements two primary logic gates for frame classification:

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
- Duplicate detection
- Classification status
- Image hash
- Notes

## ROI Configuration

### Drop Indicator (Fixed)
The "Dropped frame" text overlay appears in a fixed position at the top-left corner:
```python
ROI_DROP = {
    'x': 0,
    'y': 0,
    'width': 250,
    'height': 50
}
```

### Timecode (Auto-Detected)
The timecode display position varies per take. The system automatically detects the digital slate location on the first frame by scanning for the pattern `##:##:##:##`.

## Validation Checks

1. **Sync Status**: Verifies Camera A and B have identical Frame 0 timecodes
2. **Corruption Status**: Cross-references vision-detected corruptions with metadata flags
3. **Drop Frame Accuracy**: Compares detected physical drops against metadata counts
4. **Start Timecode**: Validates OCR Frame 0 matches metadata start timecode
5. **Indicator Health**: Reports false negatives and false positives

## Debug Mode

When `--debug` is enabled, the system saves annotated screenshots of any frame classified as:
- CORRUPTION
- DROPPED_UNDETECTED (False Negative)
- FALSE_POSITIVE
- UNDETECTED_SKIP

Screenshots are saved to `debug_frames/` with naming convention:
```
cam{A|B}_frame{NNNNNN}_{STATUS}.png
```

## Dependencies

- `opencv-python>=4.8.0` - Video processing and image analysis
- `numpy>=1.24.0` - Numerical operations
- `Pillow>=10.0.0` - Image handling
- `pytesseract>=0.3.10` - OCR engine wrapper
- `imagehash>=4.3.1` - Perceptual hashing for duplicate detection
- `pandas>=2.0.0` - Data analysis and CSV generation

## Architecture

```
video_qa_automation/
├── video_qa_validator.py   # Main application
├── test_runner.py          # Unit tests and demo
├── config.json             # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
├── qa_reports/             # Generated reports (output)
└── debug_frames/           # Flagged frame screenshots (debug)
```

### Class Hierarchy

- **Timecode**: SMPTE timecode representation and arithmetic
- **FrameAnalysis**: Per-frame analysis result container
- **CameraMetadata**: JSON metadata parser
- **FrameReader**: Video I/O and vision pipeline
- **VideoAnalyser**: Single-camera analysis engine
- **DualCameraValidator**: Orchestrates dual-stream validation

## Performance Considerations

- Frame hashing uses average hash (aHash) for speed
- Timecode ROI is locked after first-frame calibration
- Progress callbacks available for UI integration
- OCR scaling improves accuracy at cost of speed

## Troubleshooting

### OCR Failures
- Ensure Tesseract is installed and in PATH
- Check timecode display is visible and well-lit
- Adjust threshold values in config.json

### False Duplicates
- Increase `duplicate_frame_hash_distance` threshold
- Check for compression artefacts in source video

### Missing Drop Indicators
- Verify ROI_DROP coordinates match actual overlay position
- Check indicator brightness threshold

## Licence

Proprietary - Captive Devices Ltd.

## Version History

- **1.0.0** - Initial release with dual-camera validation support
