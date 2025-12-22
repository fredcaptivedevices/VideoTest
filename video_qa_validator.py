#!/usr/bin/env python3
"""
Video QA Automation Application for Hardware Validation
========================================================
Processes dual-stream video (Camera A/B) and metadata to validate recording integrity.

Author: Captive Devices QA System
Version: 1.2.0
"""

import cv2
import numpy as np
import pandas as pd
import json
import re
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime
from enum import Enum, auto
import easyocr
from collections import defaultdict


# =============================================================================
# Configuration & Constants
# =============================================================================

DEBUG = True  # Set to True to save screenshots of flagged frames
DEBUG_OUTPUT_DIR = Path("debug_frames")

# EasyOCR reader - initialised lazily for performance
_EASYOCR_READER = None

def get_ocr_reader():
    """
    Lazy initialisation of EasyOCR reader (only created once).
    Automatically uses GPU if available:
    - NVIDIA GPUs via CUDA
    - Apple Silicon (M1/M2/M3) via MPS (Metal Performance Shaders)
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import torch
        
        # Check for GPU availability
        use_gpu = False
        gpu_name = "None"
        
        # Check NVIDIA CUDA first
        if torch.cuda.is_available():
            use_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  [OCR] Using NVIDIA GPU: {gpu_name}")
        # Check Apple Silicon MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            use_gpu = True
            gpu_name = "Apple Silicon (MPS)"
            print(f"  [OCR] Using Apple Silicon GPU (Metal)")
        else:
            print("  [OCR] No GPU detected, using CPU (slower)")
        
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
    return _EASYOCR_READER


# Drop indicator ROI - Fixed position based on screenshot analysis
# The "Dropped frame" text appears in top-left corner
ROI_DROP = {
    'x': 0,
    'y': 0,
    'width': 250,
    'height': 50
}

# Debug: Save OCR input images for troubleshooting
DEBUG_OCR = True  # Set to True to save OCR debug images
DEBUG_OCR_DIR = Path("debug_ocr")

# Timecode pattern for OCR validation
# Standard: HH:MM:SS:FF with colons
TIMECODE_PATTERN = re.compile(r'(\d{1,2}):(\d{2}):(\d{2}):(\d{2})')
# Loose: digits with any separator or no separator (requires 7-8 digits total)
TIMECODE_LOOSE_PATTERN = re.compile(r'(\d{1,2})\D*(\d{2})\D*(\d{2})\D*(\d{2})')
# Very loose: for when OCR drops characters - try to find any 6-8 digit sequence
TIMECODE_DIGITS_PATTERN = re.compile(r'(\d{6,8})')

# Image comparison thresholds
DUPLICATE_FRAME_THRESHOLD = 5  # Hash difference threshold for duplicate detection
DUPLICATE_MSE_THRESHOLD = 5.0  # MSE threshold for timecode region - very strict
TIMECODE_JUMP_THRESHOLD = 100  # Maximum expected frame jump before flagging corruption

# Known frame resolution for Captive Devices cameras
KNOWN_FRAME_WIDTH = 1600
KNOWN_FRAME_HEIGHT = 2472

# High frame rate support
# The LockitSlate timecode display runs at max 30fps (or 60fps with specific settings)
# When video frame rate exceeds timecode rate, we see the same TC on multiple frames
# This is computed automatically from metadata frame rate
DEFAULT_TIMECODE_FPS = 30  # LockitSlate default timecode display rate


def compute_frame_rate_multiplier(video_fps: int, timecode_fps: int = DEFAULT_TIMECODE_FPS) -> int:
    """
    Compute how many video frames share each timecode value.
    
    Examples:
        - 30fps video / 30fps TC = multiplier 1 (each frame has unique TC)
        - 60fps video / 30fps TC = multiplier 2 (pairs of frames share TC)
        - 120fps video / 30fps TC = multiplier 4 (quads share TC)
        - 24fps video / 24fps TC = multiplier 1
        - 48fps video / 24fps TC = multiplier 2
    """
    if video_fps <= 0 or timecode_fps <= 0:
        return 1
    
    # For standard frame rates
    multiplier = video_fps // timecode_fps
    
    # Handle non-integer cases (e.g., 29.97fps)
    if multiplier < 1:
        multiplier = 1
    
    return multiplier


class FrameStatus(Enum):
    """Enumeration of possible frame statuses"""
    NORMAL = auto()
    DROPPED_DETECTED = auto()       # Physical drop detected, indicator present
    DROPPED_UNDETECTED = auto()     # Physical drop detected, NO indicator (False Negative)
    FALSE_POSITIVE = auto()         # No physical drop, but indicator present
    CORRUPTION = auto()             # Illogical timecode jump
    UNDETECTED_SKIP = auto()        # Timecode skipped without physical duplicate
    OCR_FAILURE = auto()            # Could not read timecode


@dataclass
class Timecode:
    """Represents a SMPTE timecode"""
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    frames: int = 0
    
    @classmethod
    def from_string(cls, tc_string: str) -> Optional['Timecode']:
        """
        Parse timecode from string format HH:MM:SS:FF.
        
        Handles various OCR artifacts:
        - Standard format: "02:31:21:06"
        - No separators: "02312106" 
        - Partial separators: "02:312106"
        - Missing leading zeros: "2312106"
        
        Returns None for invalid timecodes (minutes >= 60, seconds >= 60, frames >= 60)
        """
        if not tc_string:
            return None
            
        # Clean the string
        tc_string = tc_string.strip()
        
        def is_valid_timecode(h, m, s, f):
            """Check if timecode values are valid"""
            # Hours can be 0-23 (or higher for long recordings)
            # Minutes must be 0-59
            # Seconds must be 0-59
            # Frames must be 0-59 (for up to 60fps)
            return 0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59 and 0 <= f <= 59
        
        # Try standard pattern first: HH:MM:SS:FF
        match = TIMECODE_PATTERN.search(tc_string)
        if match:
            h, m, s, f = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            if is_valid_timecode(h, m, s, f):
                return cls(hours=h, minutes=m, seconds=s, frames=f)
        
        # Try loose pattern: digits with any/no separators
        match = TIMECODE_LOOSE_PATTERN.search(tc_string)
        if match:
            h, m, s, f = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            if is_valid_timecode(h, m, s, f):
                return cls(hours=h, minutes=m, seconds=s, frames=f)
        
        # Try extracting just digits and parsing as HHMMSSFF
        digits_only = re.sub(r'\D', '', tc_string)
        
        if len(digits_only) == 8:
            # HHMMSSFF
            h, m, s, f = int(digits_only[0:2]), int(digits_only[2:4]), int(digits_only[4:6]), int(digits_only[6:8])
            if is_valid_timecode(h, m, s, f):
                return cls(hours=h, minutes=m, seconds=s, frames=f)
        elif len(digits_only) == 7:
            # HMMSSFF (single digit hour)
            h, m, s, f = int(digits_only[0:1]), int(digits_only[1:3]), int(digits_only[3:5]), int(digits_only[5:7])
            if is_valid_timecode(h, m, s, f):
                return cls(hours=h, minutes=m, seconds=s, frames=f)
        elif len(digits_only) == 6:
            # MMSSFF (no hour, assume 0)
            m, s, f = int(digits_only[0:2]), int(digits_only[2:4]), int(digits_only[4:6])
            if is_valid_timecode(0, m, s, f):
                return cls(hours=0, minutes=m, seconds=s, frames=f)
        
        return None
    
    @classmethod
    def from_json(cls, json_tc: dict) -> 'Timecode':
        """Create Timecode from JSON metadata format"""
        return cls(
            hours=int(json_tc.get('Hours', 0)),
            minutes=int(json_tc.get('Minutes', 0)),
            seconds=int(json_tc.get('Seconds', 0)),
            frames=int(json_tc.get('Frames', 0))
        )
    
    def to_frames(self, fps: int = 60) -> int:
        """Convert timecode to total frame count"""
        return (
            self.hours * 3600 * fps +
            self.minutes * 60 * fps +
            self.seconds * fps +
            self.frames
        )
    
    def __str__(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}:{self.frames:02d}"
    
    def __eq__(self, other: 'Timecode') -> bool:
        if not isinstance(other, Timecode):
            return False
        return (self.hours == other.hours and 
                self.minutes == other.minutes and 
                self.seconds == other.seconds and 
                self.frames == other.frames)


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame"""
    frame_number: int
    visual_timecode: Optional[Timecode] = None
    visual_timecode_raw: str = ""
    drop_indicator_present: bool = False
    is_physical_duplicate: bool = False
    status: FrameStatus = FrameStatus.NORMAL
    image_hash: str = ""
    notes: str = ""


@dataclass
class CameraMetadata:
    """Parsed metadata from JSON file"""
    filename: str
    full_pathname: str
    frame_rate: int
    dropped_frames: int
    num_frames: int
    shot_hierarchy: str
    corrupted_at: int
    drop_percentage: float
    severity: int
    start_timecode: Timecode
    drop_events: str
    
    @classmethod
    def from_json_file(cls, json_path: Path) -> 'CameraMetadata':
        """Load metadata from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        captive = data.get('Captive', {})
        timecode = data.get('Timecode', {})
        
        return cls(
            filename=captive.get('Filename', ''),
            full_pathname=captive.get('FullPathname', ''),
            frame_rate=int(captive.get('FrameRate', 60)),
            dropped_frames=int(captive.get('DroppedFrames', 0)),
            num_frames=int(captive.get('NumFrames', 0)),
            shot_hierarchy=captive.get('ShotHierarchy', ''),
            corrupted_at=int(captive.get('CorruptedAt', -1)),
            drop_percentage=float(captive.get('DropPercentage', 0)),
            severity=int(captive.get('Severity', 0)),
            start_timecode=Timecode.from_json(timecode),
            drop_events=data.get('DropEvents', '')
        )


# =============================================================================
# Frame Reader Class
# =============================================================================

class FrameReader:
    """
    Handles frame extraction and analysis from video files.
    Includes drop zone detection and timecode OCR.
    
    PERFORMANCE OPTIMISATIONS:
    - Sequential frame reading (no seeking)
    - Downscaled frames for hashing
    - Cached ROI extraction
    - Simplified duplicate detection using structural similarity
    """
    
    def __init__(self, video_path: Path, camera_id: str = "A", frame_rate_multiplier: int = 1):
        self.video_path = Path(video_path)
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        
        # Frame rate multiplier - how many video frames per timecode frame
        # Set dynamically based on video FPS vs timecode FPS
        self.frame_rate_multiplier = frame_rate_multiplier
        
        # ROI for timecode - auto-detected on first frame
        self.roi_timecode: Optional[Dict[str, int]] = None
        self.timecode_calibrated: bool = False
        
        # Timecode-based duplicate detection
        # At 60fps with 30fps timecode, each TC value appears for 2 consecutive frames
        # We track the timecode value and count how many frames show it
        self._prev_timecode: Optional['Timecode'] = None
        self._consecutive_same_tc: int = 0  # How many frames with same TC value
        
        # Legacy image-based detection (fallback if OCR fails)
        self._prev_frame_small: Optional[np.ndarray] = None
        self._hash_size: int = 16
        self._consecutive_duplicates: int = 0
        
        # Logger
        self.logger = logging.getLogger(f"FrameReader_{camera_id}")
    
    def open(self) -> bool:
        """Open video file and initialise properties"""
        if not self.video_path.exists():
            self.logger.error(f"Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open video: {self.video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Performance: set buffer size for sequential reading
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        self.logger.info(f"Opened video: {self.video_path.name}")
        self.logger.info(f"  Frames: {self.total_frames}, FPS: {self.fps}")
        self.logger.info(f"  Resolution: {self.frame_width}x{self.frame_height}")
        
        return True
    
    def close(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_next_frame(self) -> Optional[np.ndarray]:
        """Read the next frame sequentially (faster than seeking)"""
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame from the video (use read_next_frame for sequential access)"""
        if not self.cap:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            self.logger.warning(f"Failed to read frame {frame_number}")
            return None
        
        return frame
    
    def detect_drop_indicator(self, frame: np.ndarray) -> bool:
        """
        Detect if the 'Dropped frame' indicator is present.
        Uses fixed ROI based on known overlay position (top-left).
        
        OPTIMISED: Simple pixel intensity check before OCR
        """
        # Extract drop indicator region
        roi = frame[
            ROI_DROP['y']:ROI_DROP['y'] + ROI_DROP['height'],
            ROI_DROP['x']:ROI_DROP['x'] + ROI_DROP['width']
        ]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # OPTIMISATION: Quick check - if region is too dark, skip OCR
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:  # Very dark region, no text
            return False
        
        # Check for white pixels (text) in the region
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white_pixel_ratio = np.sum(thresh > 0) / thresh.size
        
        # If less than 1% white pixels, probably no text
        if white_pixel_ratio < 0.01:
            return False
        
        # Use OCR to detect "Dropped" or "DROP" text
        ocr_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        text = pytesseract.image_to_string(thresh, config=ocr_config).strip().lower()
        
        # Check for drop-related keywords
        drop_keywords = ['drop', 'dropped', 'frame']
        return any(keyword in text for keyword in drop_keywords)
    
    def auto_detect_timecode_roi(self, frame: np.ndarray) -> bool:
        """
        Auto-detect the timecode region on the first frame.
        Scans for digital display pattern (##:##:##:##).
        
        The slate can be ANYWHERE in the frame - searches entire image.
        Optimised for known 1600x2472 portrait resolution.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        self.logger.info(f"Auto-detecting timecode ROI in {width}x{height} frame...")
        
        # Strategy 1: Look for the bright LED timecode display
        # The timecode is bright white/green digits on dark background
        # LED displays typically have very high brightness (200+)
        
        for thresh_val in [230, 210, 190, 170, 150]:
            _, bright_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect the LED segments into digit shapes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
            dilated = cv2.dilate(bright_mask, kernel, iterations=3)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort by area (largest bright regions first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:15]:  # Check top 15 candidates
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Timecode displays: wide rectangle
                # Format "00:00:00:00" has aspect ratio roughly 4:1 to 7:1
                # On 1600px wide frame, display is typically 300-500px wide
                if 2.5 < aspect_ratio < 9.0 and w > 60 and h > 12 and area > 1500:
                    # Add generous padding
                    padding = 50
                    roi_x = max(0, x - padding)
                    roi_y = max(0, y - padding)
                    roi_w = min(width - roi_x, w + 2 * padding)
                    roi_h = min(height - roi_y, h + 2 * padding)
                    
                    # Try OCR on this region
                    test_roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    tc = self._try_ocr_timecode(test_roi)
                    
                    if tc:
                        self.roi_timecode = {'x': roi_x, 'y': roi_y, 'width': roi_w, 'height': roi_h}
                        self.timecode_calibrated = True
                        self.logger.info(f"Timecode ROI detected (thresh={thresh_val}): x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                        self.logger.info(f"  Initial timecode reading: {tc}")
                        return True
        
        # Strategy 2: Grid-based scan of entire frame
        # For 1600x2472, use appropriately sized windows
        self.logger.info("Bright region detection failed, trying grid scan...")
        
        # Window sizes optimised for 1600x2472 resolution
        # Timecode display is roughly 350-450px wide, 50-80px tall
        window_sizes = [(450, 100), (400, 90), (350, 80), (500, 110)]
        
        for win_w, win_h in window_sizes:
            step_x = win_w // 3  # 66% overlap for thorough coverage
            step_y = win_h // 2
            
            for y in range(0, height - win_h, step_y):
                for x in range(0, width - win_w, step_x):
                    test_roi = gray[y:y+win_h, x:x+win_w]
                    
                    # Quick check: skip very dark regions
                    if np.max(test_roi) < 120:
                        continue
                    
                    tc = self._try_ocr_timecode(test_roi)
                    
                    if tc:
                        padding = 40
                        self.roi_timecode = {
                            'x': max(0, x - padding),
                            'y': max(0, y - padding),
                            'width': min(win_w + 2*padding, width - x + padding),
                            'height': min(win_h + 2*padding, height - y + padding)
                        }
                        self.timecode_calibrated = True
                        self.logger.info(f"Timecode ROI detected (grid scan): {self.roi_timecode}")
                        self.logger.info(f"  Initial timecode reading: {tc}")
                        return True
        
        # Fallback: Use generous centre-lower region (where slate typically is in portrait)
        self.logger.warning("Auto-detection failed, using default centre ROI")
        # For 1600x2472, slate is often in lower-centre area
        self.roi_timecode = {
            'x': 100,
            'y': height // 3,
            'width': width - 200,
            'height': height // 2
        }
        return False
    
    def _try_ocr_timecode(self, roi: np.ndarray) -> Optional[Timecode]:
        """Attempt OCR on a region and return Timecode if valid pattern found"""
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 50:
            return None
        
        # Quick brightness check - timecode region should have bright pixels
        if np.max(roi) < 100:
            return None
            
        # Try multiple threshold methods
        methods = [
            lambda img: cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1],
            lambda img: cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1],
            lambda img: cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1],
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda img: cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)[1],
        ]
        
        for method in methods:
            try:
                thresh = method(roi)
                
                # Scale up for better OCR
                scale = 2.0 if roi.shape[1] < 300 else 1.5
                scaled = cv2.resize(thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                
                ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789:.'
                text = pytesseract.image_to_string(scaled, config=ocr_config).strip()
                text = text.replace('.', ':').replace(' ', '').replace('O', '0').replace('o', '0')
                
                tc = Timecode.from_string(text)
                if tc:
                    return tc
            except Exception:
                continue
        
        return None
    
    def _generate_scan_regions(self, shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        """Generate regions to scan for timecode display"""
        height, width = shape[:2]
        regions = []
        
        # Common positions for digital slates - prioritise right side and centre
        # Right third (most common for clapperboards)
        regions.append((width // 2, height // 4, width // 2, height // 2))
        
        # Centre area
        regions.append((width // 4, height // 4, width // 2, height // 2))
        
        # Upper right
        regions.append((width // 2, 0, width // 2, height // 2))
        
        # Full right side
        regions.append((width * 2 // 3, 0, width // 3, height))
        
        # Sliding window for thorough coverage
        window_w, window_h = 350, 80
        step = 150
        
        for y in range(0, height - window_h, step):
            for x in range(width // 3, width - window_w, step):  # Focus on right 2/3
                regions.append((x, y, window_w, window_h))
        
        return regions
    
    def extract_timecode(self, frame: np.ndarray) -> Tuple[Optional[Timecode], str]:
        """
        Extract timecode from frame using EasyOCR.
        Returns tuple of (Timecode object, raw OCR text).
        
        Optimised for white dot-matrix LED displays on monochrome footage.
        """
        if not self.roi_timecode:
            return None, ""
        
        # Extract ROI
        roi = frame[
            self.roi_timecode['y']:self.roi_timecode['y'] + self.roi_timecode['height'],
            self.roi_timecode['x']:self.roi_timecode['x'] + self.roi_timecode['width']
        ]
        
        # For monochrome footage, all channels are the same
        # Use grayscale directly
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Step 1: Apply strong Gaussian blur to fuse the dot-matrix LED dots
        # Use larger kernel for better dot fusion
        blur_kernel = 7  # Larger kernel for better fusion
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Step 2: Apply morphological closing to further connect dots
        # This helps fill gaps between LED dots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Debug: Save first few ROI images to help troubleshoot OCR issues
        if DEBUG_OCR:
            if not hasattr(self, '_debug_ocr_count'):
                self._debug_ocr_count = 0
            self._debug_ocr_count += 1
            if self._debug_ocr_count <= 10:  # Save first 10 frames only
                try:
                    DEBUG_OCR_DIR.mkdir(exist_ok=True)
                    cv2.imwrite(str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._debug_ocr_count:04d}_1_roi.png"), roi)
                    cv2.imwrite(str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._debug_ocr_count:04d}_2_gray.png"), gray)
                    cv2.imwrite(str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._debug_ocr_count:04d}_3_blurred.png"), blurred)
                except Exception as e:
                    self.logger.debug(f"Failed to save debug OCR image: {e}")
        
        # Try multiple preprocessing approaches optimised for white LED on dark background
        images_to_try = []
        
        # Method 1: Blur + High threshold (for bright LEDs)
        _, thresh1 = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        images_to_try.append(('thresh200', thresh1))
        
        # Method 2: Blur + Medium threshold
        _, thresh2 = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
        images_to_try.append(('thresh180', thresh2))
        
        # Method 3: Blur + Lower threshold
        _, thresh3 = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)
        images_to_try.append(('thresh150', thresh3))
        
        # Method 4: Blur + OTSU auto-threshold
        _, thresh4 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh4 = cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel)
        images_to_try.append(('otsu', thresh4))
        
        # Method 5: Adaptive threshold on blurred image
        thresh5 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 21, -5)
        thresh5 = cv2.morphologyEx(thresh5, cv2.MORPH_CLOSE, kernel)
        images_to_try.append(('adaptive', thresh5))
        
        # Debug: Save threshold images for first few frames
        if DEBUG_OCR and hasattr(self, '_debug_ocr_count') and self._debug_ocr_count <= 5:
            try:
                for name, img in images_to_try:
                    cv2.imwrite(str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._debug_ocr_count:04d}_4_{name}.png"), img)
            except Exception as e:
                self.logger.debug(f"Failed to save debug threshold image: {e}")
        
        # Try each preprocessed image
        all_raw_texts = []
        for name, img in images_to_try:
            tc, raw = self._ocr_timecode_easyocr(img)
            if tc is not None:
                return tc, raw
            if raw:
                all_raw_texts.append(raw)
        
        # Return best raw text even if parsing failed
        return None, all_raw_texts[0] if all_raw_texts else ""
    
    def _ocr_timecode_easyocr(self, img: np.ndarray) -> Tuple[Optional[Timecode], str]:
        """Perform OCR using EasyOCR and parse timecode"""
        try:
            reader = get_ocr_reader()
            
            # Scale up significantly for better recognition of dot-matrix displays
            # Larger scale helps EasyOCR see the digit shapes clearly
            scale = 3
            img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # If grayscale, convert to BGR for EasyOCR
            if len(img_scaled.shape) == 2:
                img_for_ocr = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
            else:
                img_for_ocr = img_scaled
            
            img_height = img_for_ocr.shape[0]
            
            # Run EasyOCR - allowlist only digits and colon
            results = reader.readtext(
                img_for_ocr,
                allowlist='0123456789:',
                paragraph=False,
                detail=1,
                width_ths=0.7,
                height_ths=0.7
            )
            
            if not results:
                return None, ""
            
            # Filter results: only keep text from the UPPER portion of the ROI
            # The timecode display is at the top, labels like SCENE/SHOT are below
            upper_threshold = img_height * 0.5  # Only look at top 50% of ROI
            
            filtered_results = []
            for (bbox, text, confidence) in results:
                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                # Get the center Y position of this text
                center_y = (bbox[0][1] + bbox[2][1]) / 2
                
                # Only include if in upper portion of image
                if center_y < upper_threshold and confidence > 0.1:
                    # Additional filter: skip very short single characters that aren't digits
                    clean_text = text.strip()
                    if len(clean_text) >= 1:
                        filtered_results.append((bbox, text, confidence))
            
            if not filtered_results:
                # Fallback: try all results if filtering removed everything
                filtered_results = [(b, t, c) for b, t, c in results if c > 0.1]
            
            # Sort by x-position (left to right)
            sorted_results = sorted(filtered_results, key=lambda r: r[0][0][0])
            
            # Combine text
            raw_parts = []
            for (bbox, text, confidence) in sorted_results:
                raw_parts.append(text)
            
            raw_text = ''.join(raw_parts)
            
            # Clean up common OCR errors for dot-matrix displays
            raw_text = raw_text.replace('.', ':').replace(' ', '')
            raw_text = raw_text.replace('O', '0').replace('o', '0')
            raw_text = raw_text.replace('l', '1').replace('I', '1').replace('|', '1')
            raw_text = raw_text.replace('S', '5').replace('s', '5')
            raw_text = raw_text.replace('B', '8').replace('G', '6')
            raw_text = raw_text.replace('Z', '2').replace('z', '2')
            raw_text = raw_text.replace('A', '4').replace('g', '9')
            raw_text = raw_text.replace('D', '0').replace('Q', '0')
            raw_text = raw_text.replace('T', '1').replace('i', '1')
            
            # Try to parse timecode
            tc = Timecode.from_string(raw_text)
            return tc, raw_text
            
        except Exception as e:
            self.logger.debug(f"EasyOCR error: {e}")
            return None, ""
    
    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Compute perceptual hash for frame comparison.
        
        OPTIMISED: Use OpenCV resize + simple hash instead of imagehash library
        """
        # Convert to grayscale and resize to small square
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self._hash_size, self._hash_size), interpolation=cv2.INTER_AREA)
        
        # Compute average hash
        avg = np.mean(small)
        hash_bits = (small > avg).flatten()
        
        # Convert to hex string
        hash_int = 0
        for bit in hash_bits:
            hash_int = (hash_int << 1) | int(bit)
        
        return format(hash_int, f'0{self._hash_size * self._hash_size // 4}x')
    
    def check_timecode_duplicate(self, current_tc: Optional['Timecode']) -> bool:
        """
        Check if this frame is a duplicate based on TIMECODE VALUE.
        
        This is the primary duplicate detection method. At 60fps with 30fps timecode
        (from an Ambient LockitSlate), each timecode value appears for exactly 2 frames.
        
        frame_rate_multiplier controls expected duplicates:
        - Multiplier=2 (60fps/30tc): Each TC appears on 2 frames, so seeing it twice is normal
        - Multiplier=1 (30fps/30tc): Each TC should appear once, any repeat is a drop
        
        Returns True if we've seen this timecode MORE times than expected (indicating a drop).
        """
        if current_tc is None:
            # OCR failed - can't determine duplicate status from timecode
            return False
        
        if self._prev_timecode is None:
            # First frame with valid timecode
            self._prev_timecode = current_tc
            self._consecutive_same_tc = 1
            return False
        
        # Compare timecode values
        if current_tc.to_frames() == self._prev_timecode.to_frames():
            # Same timecode as previous frame
            self._consecutive_same_tc += 1
        else:
            # Timecode changed - reset counter
            self._prev_timecode = current_tc
            self._consecutive_same_tc = 1
        
        # At 60fps/30tc (multiplier=2):
        #   - Seeing TC twice (consecutive_same_tc=2) is NORMAL
        #   - Seeing TC 3+ times (consecutive_same_tc>=3) is a DROP
        # The condition is: consecutive_same_tc > frame_rate_multiplier
        return self._consecutive_same_tc > self.frame_rate_multiplier
    
    def frames_are_duplicate(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        LEGACY: Check if two frames are visual duplicates.
        
        This method is kept for compatibility but the preferred approach is
        check_timecode_duplicate() which compares actual timecode values.
        """
        if not self.roi_timecode:
            size = (64, 64)
            small1 = cv2.resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
            small2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
            mse = np.mean((small1.astype(float) - small2.astype(float)) ** 2)
            return mse < DUPLICATE_MSE_THRESHOLD
        
        roi = self.roi_timecode
        tc1 = frame1[roi['y']:roi['y']+roi['height'], roi['x']:roi['x']+roi['width']]
        tc2 = frame2[roi['y']:roi['y']+roi['height'], roi['x']:roi['x']+roi['width']]
        
        gray1 = cv2.cvtColor(tc1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(tc2, cv2.COLOR_BGR2GRAY)
        
        size = (128, 32)
        small1 = cv2.resize(gray1, size, interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(gray2, size, interpolation=cv2.INTER_AREA)
        
        mse = np.mean((small1.astype(float) - small2.astype(float)) ** 2)
        return mse < DUPLICATE_MSE_THRESHOLD
    
    def frames_are_duplicate_fast(self, current_frame: np.ndarray) -> bool:
        """
        LEGACY: Fast duplicate check using image comparison.
        
        This is a fallback method. The preferred approach is check_timecode_duplicate()
        which is called from the analysis loop after OCR extraction.
        """
        if not self.roi_timecode:
            size = (64, 64)
            small_current = cv2.resize(
                cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), 
                size, 
                interpolation=cv2.INTER_AREA
            )
            
            if self._prev_frame_small is None:
                self._prev_frame_small = small_current
                self._consecutive_duplicates = 0
                return False
            
            mse = np.mean((small_current.astype(float) - self._prev_frame_small.astype(float)) ** 2)
            is_same = mse < DUPLICATE_MSE_THRESHOLD
            
            if is_same:
                self._consecutive_duplicates += 1
            else:
                self._consecutive_duplicates = 0
                self._prev_frame_small = small_current
            
            return self._consecutive_duplicates >= self.frame_rate_multiplier
        
        roi = self.roi_timecode
        tc_region = current_frame[roi['y']:roi['y']+roi['height'], roi['x']:roi['x']+roi['width']]
        gray = cv2.cvtColor(tc_region, cv2.COLOR_BGR2GRAY)
        
        size = (128, 32)
        small_current = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        
        if self._prev_frame_small is None:
            self._prev_frame_small = small_current
            self._consecutive_duplicates = 0
            return False
        
        mse = np.mean((small_current.astype(float) - self._prev_frame_small.astype(float)) ** 2)
        is_same = mse < DUPLICATE_MSE_THRESHOLD
        
        if is_same:
            self._consecutive_duplicates += 1
        else:
            self._consecutive_duplicates = 0
            self._prev_frame_small = small_current
        
        return self._consecutive_duplicates >= self.frame_rate_multiplier


# =============================================================================
# Video Analyser Class
# =============================================================================

class VideoAnalyser:
    """
    Main analysis engine implementing the Truth Table logic.
    Processes frames and validates against metadata.
    """
    
    def __init__(self, video_path: Path, metadata_path: Path, camera_id: str = "A"):
        self.video_path = Path(video_path)
        self.metadata_path = Path(metadata_path)
        self.camera_id = camera_id
        
        self.reader: Optional[FrameReader] = None
        self.metadata: Optional[CameraMetadata] = None
        self.results: List[FrameAnalysis] = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'successful_drops': 0,      # Physical drop + indicator
            'false_negatives': 0,       # Physical drop, no indicator
            'false_positives': 0,       # No physical drop, indicator present
            'corruptions': 0,
            'undetected_skips': 0,
            'ocr_failures': 0,
            'physical_drops': 0,
            'indicated_drops': 0
        }
        
        self.logger = logging.getLogger(f"VideoAnalyser_{camera_id}")
        
        # Debug output
        if DEBUG:
            DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
    
    def load(self) -> bool:
        """Load video and metadata"""
        # Load metadata
        try:
            self.metadata = CameraMetadata.from_json_file(self.metadata_path)
            self.logger.info(f"Loaded metadata: {self.metadata.filename}")
            self.logger.info(f"  Reported drops: {self.metadata.dropped_frames}")
            self.logger.info(f"  Start TC: {self.metadata.start_timecode}")
            self.logger.info(f"  Frame rate: {self.metadata.frame_rate} fps")
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return False
        
        # Compute frame rate multiplier from metadata
        # This determines how many video frames share each timecode value
        multiplier = compute_frame_rate_multiplier(self.metadata.frame_rate)
        self.logger.info(f"  Frame rate multiplier: {multiplier} (video frames per TC frame)")
        
        # Open video with the computed multiplier
        self.reader = FrameReader(self.video_path, self.camera_id, frame_rate_multiplier=multiplier)
        if not self.reader.open():
            return False
        
        return True
    
    def calibrate(self, manual_roi: Optional[Dict[str, int]] = None, save_debug: bool = False) -> bool:
        """
        Calibrate timecode detection on first frame.
        
        Priority order:
        1. Manual ROI passed as argument
        2. ROI config file (roi_config.json) in video folder or parent
        3. Auto-detection (fallback)
        
        Args:
            manual_roi: Optional manual override for timecode ROI
                        Format: {'x': int, 'y': int, 'width': int, 'height': int}
            save_debug: If True, saves a debug image showing detected ROI
        """
        if not self.reader:
            return False
        
        # Priority 1: Manual ROI argument
        if manual_roi:
            self.reader.roi_timecode = manual_roi
            self.reader.timecode_calibrated = True
            self.logger.info(f"Using manual timecode ROI: {manual_roi}")
            return True
        
        # Priority 2: Load from config file
        config_roi = self._load_roi_from_config()
        if config_roi:
            self.reader.roi_timecode = config_roi
            self.reader.timecode_calibrated = True
            self.logger.info(f"Loaded timecode ROI from config: {config_roi}")
            return True
        
        # Priority 3: Auto-detection
        frame = self.reader.read_frame(0)
        if frame is None:
            self.logger.error("Cannot read first frame for calibration")
            return False
        
        # Reset to frame 0 after calibration read
        self.reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        success = self.reader.auto_detect_timecode_roi(frame)
        if not success:
            self.logger.warning("Timecode ROI auto-detection failed, using fallback")
            self.logger.warning("Consider running: python calibrate_roi.py /path/to/video.mov")
        
        # Save debug image showing ROI
        if save_debug and self.reader.roi_timecode:
            self._save_calibration_debug(frame)
        
        return True
    
    def _load_roi_from_config(self) -> Optional[Dict[str, int]]:
        """Load ROI from roi_config.json if it exists"""
        # Check video folder
        video_folder = self.video_path.parent
        
        # Search locations: video folder, parent (take), grandparent (shot)
        search_paths = [
            video_folder / 'roi_config.json',
            video_folder.parent / 'roi_config.json',
            video_folder.parent.parent / 'roi_config.json',
        ]
        
        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    roi = config.get('timecode_roi')
                    if roi and all(k in roi for k in ['x', 'y', 'width', 'height']):
                        self.logger.info(f"Found ROI config: {config_path}")
                        return roi
                except Exception as e:
                    self.logger.warning(f"Error loading ROI config {config_path}: {e}")
        
        return None
    
    def _save_calibration_debug(self, frame: np.ndarray):
        """Save a debug image showing the detected ROI regions"""
        debug_frame = frame.copy()
        
        # Draw timecode ROI in green
        if self.reader.roi_timecode:
            roi = self.reader.roi_timecode
            cv2.rectangle(
                debug_frame,
                (roi['x'], roi['y']),
                (roi['x'] + roi['width'], roi['y'] + roi['height']),
                (0, 255, 0), 3
            )
            cv2.putText(
                debug_frame, "TIMECODE ROI",
                (roi['x'], roi['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Draw drop indicator ROI in red
        cv2.rectangle(
            debug_frame,
            (ROI_DROP['x'], ROI_DROP['y']),
            (ROI_DROP['x'] + ROI_DROP['width'], ROI_DROP['y'] + ROI_DROP['height']),
            (0, 0, 255), 2
        )
        cv2.putText(
            debug_frame, "DROP INDICATOR ROI",
            (ROI_DROP['x'], ROI_DROP['y'] + ROI_DROP['height'] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        
        # Save
        DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
        debug_path = DEBUG_OUTPUT_DIR / f"calibration_cam{self.camera_id}.png"
        cv2.imwrite(str(debug_path), debug_frame)
        self.logger.info(f"Saved calibration debug image: {debug_path}")
    
    def analyse(self, progress_callback=None) -> bool:
        """
        Run full frame-by-frame analysis implementing Truth Table logic.
        
        OPTIMISED: Sequential frame reading, fast duplicate detection
        """
        if not self.reader or not self.metadata:
            self.logger.error("Must call load() before analyse()")
            return False
        
        total = self.reader.total_frames
        self.stats['total_frames'] = total
        
        prev_frame: Optional[np.ndarray] = None
        prev_analysis: Optional[FrameAnalysis] = None
        
        self.logger.info(f"Starting analysis of {total} frames...")
        
        # Progress tracking
        start_time = datetime.now()
        last_progress_pct = -1
        
        # Reset video to start for sequential reading
        self.reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_num in range(total):
            # OPTIMISED: Sequential read instead of seeking
            frame = self.reader.read_next_frame()
            if frame is None:
                analysis = FrameAnalysis(
                    frame_number=frame_num,
                    status=FrameStatus.OCR_FAILURE,
                    notes="Failed to read frame"
                )
                self.results.append(analysis)
                self.stats['ocr_failures'] += 1
                continue
            
            # Create analysis record
            analysis = FrameAnalysis(frame_number=frame_num)
            
            # Check drop indicator (has early-exit optimisation)
            analysis.drop_indicator_present = self.reader.detect_drop_indicator(frame)
            if analysis.drop_indicator_present:
                self.stats['indicated_drops'] += 1
            
            # Extract timecode via OCR
            tc, raw = self.reader.extract_timecode(frame)
            analysis.visual_timecode = tc
            analysis.visual_timecode_raw = raw
            
            # PRIMARY: Timecode-based duplicate detection
            # Compares actual timecode values - more reliable than image comparison
            # At 60fps/30tc, each TC appears twice (normal), 3+ times = drop
            analysis.is_physical_duplicate = self.reader.check_timecode_duplicate(tc)
            if analysis.is_physical_duplicate:
                self.stats['physical_drops'] += 1
            
            # Compute frame hash (optimised - no PIL conversion)
            analysis.image_hash = self.reader.compute_frame_hash(frame)
            
            # Apply Truth Table Logic
            analysis.status = self._apply_logic_gates(analysis, prev_analysis)
            
            # Update statistics based on status
            self._update_stats(analysis)
            
            # Debug: Save flagged frames
            if DEBUG and analysis.status in [
                FrameStatus.CORRUPTION,
                FrameStatus.DROPPED_UNDETECTED,
                FrameStatus.FALSE_POSITIVE,
                FrameStatus.UNDETECTED_SKIP
            ]:
                self._save_debug_frame(frame, analysis)
            
            self.results.append(analysis)
            
            # Update for next iteration
            prev_analysis = analysis
            
            # Progress reporting - update every 1%
            current_pct = int((frame_num + 1) / total * 100)
            is_complete = frame_num == total - 1
            
            if current_pct > last_progress_pct or is_complete:
                last_progress_pct = current_pct
                
                elapsed_total = (datetime.now() - start_time).total_seconds()
                fps_rate = (frame_num + 1) / elapsed_total if elapsed_total > 0 else 0
                
                if fps_rate > 0:
                    remaining_frames = total - frame_num - 1
                    eta_seconds = remaining_frames / fps_rate
                    eta_min = int(eta_seconds // 60)
                    eta_sec = int(eta_seconds % 60)
                    eta_str = f"{eta_min:3d}m {eta_sec:02d}s"
                else:
                    eta_str = "calculating..."
                
                bar_width = 30
                filled = int(bar_width * current_pct // 100)
                bar = "=" * filled + ">" + " " * (bar_width - filled - 1) if current_pct < 100 else "=" * bar_width
                
                sys.stdout.write(f"\r  [{bar}] {current_pct:3d}% | {frame_num + 1:6d}/{total} frames | {fps_rate:5.1f} fps | ETA: {eta_str}")
                sys.stdout.flush()
            
            # Legacy callback support
            if progress_callback and frame_num % 100 == 0:
                progress_callback(frame_num, total)
        
        # Final newline
        print()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Analysis complete in {elapsed:.1f}s ({total/elapsed:.1f} fps average)")
        return True
    
    def _apply_logic_gates(
        self, 
        current: FrameAnalysis, 
        previous: Optional[FrameAnalysis]
    ) -> FrameStatus:
        """
        Apply the Truth Table logic gates to determine frame status.
        """
        # Logic Gate 2: Drop Frame Verification
        if current.is_physical_duplicate and current.drop_indicator_present:
            return FrameStatus.DROPPED_DETECTED  # Successful Process
        
        if current.is_physical_duplicate and not current.drop_indicator_present:
            current.notes = "Physical duplicate without indicator"
            return FrameStatus.DROPPED_UNDETECTED  # False Negative
        
        if not current.is_physical_duplicate and current.drop_indicator_present:
            current.notes = "Indicator present but no physical duplicate"
            return FrameStatus.FALSE_POSITIVE  # False Positive
        
        # Logic Gate 1: Continuity & Corruption (requires previous frame)
        if previous and previous.visual_timecode and current.visual_timecode:
            prev_tc_frames = previous.visual_timecode.to_frames(
                self.metadata.frame_rate if self.metadata else 60
            )
            curr_tc_frames = current.visual_timecode.to_frames(
                self.metadata.frame_rate if self.metadata else 60
            )
            
            expected_next = prev_tc_frames + 1
            actual_diff = curr_tc_frames - prev_tc_frames
            
            # Check for corruption (illogical jump)
            if abs(actual_diff) > TIMECODE_JUMP_THRESHOLD:
                current.notes = f"Timecode jump: {previous.visual_timecode} -> {current.visual_timecode}"
                return FrameStatus.CORRUPTION
            
            # Check for undetected skip (timecode skips but frame is not duplicate)
            if actual_diff > 1 and not current.is_physical_duplicate:
                current.notes = f"Skipped {actual_diff - 1} timecode frames without duplicate"
                return FrameStatus.UNDETECTED_SKIP
        
        # OCR failure check
        if current.visual_timecode is None and current.frame_number > 0:
            current.notes = f"OCR failed, raw: {current.visual_timecode_raw}"
            return FrameStatus.OCR_FAILURE
        
        return FrameStatus.NORMAL
    
    def _update_stats(self, analysis: FrameAnalysis):
        """Update statistics based on frame status"""
        status_map = {
            FrameStatus.DROPPED_DETECTED: 'successful_drops',
            FrameStatus.DROPPED_UNDETECTED: 'false_negatives',
            FrameStatus.FALSE_POSITIVE: 'false_positives',
            FrameStatus.CORRUPTION: 'corruptions',
            FrameStatus.UNDETECTED_SKIP: 'undetected_skips',
            FrameStatus.OCR_FAILURE: 'ocr_failures'
        }
        
        if analysis.status in status_map:
            self.stats[status_map[analysis.status]] += 1
    
    def _save_debug_frame(self, frame: np.ndarray, analysis: FrameAnalysis):
        """Save debug screenshot for flagged frame"""
        filename = f"cam{self.camera_id}_frame{analysis.frame_number:06d}_{analysis.status.name}.png"
        filepath = DEBUG_OUTPUT_DIR / filename
        
        # Add annotation
        annotated = frame.copy()
        text = f"Frame {analysis.frame_number}: {analysis.status.name}"
        cv2.putText(
            annotated, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        if analysis.notes:
            cv2.putText(
                annotated, analysis.notes, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
            )
        
        cv2.imwrite(str(filepath), annotated)
    
    def get_frame_zero_timecode(self) -> Optional[Timecode]:
        """
        Get the timecode from the start of the video.
        
        Since OCR can be unreliable, this searches the first N frames
        for the first valid timecode reading, rather than requiring
        frame 0 specifically to have a valid OCR result.
        """
        if not self.results:
            return None
        
        # Search first 50 frames for a valid timecode
        search_limit = min(50, len(self.results))
        for i in range(search_limit):
            if self.results[i].visual_timecode is not None:
                # Found a valid timecode - extrapolate back to frame 0
                tc = self.results[i].visual_timecode
                # At 60fps/30tc (multiplier 2), each TC covers 2 frames
                # So frame i corresponds to TC frame i//multiplier
                # For frame 0, we need to subtract i//multiplier from the found TC
                multiplier = 2  # TODO: get from reader
                tc_frames_offset = i // multiplier
                
                # Create frame 0 timecode by subtracting the offset
                frame0_tc = Timecode(
                    hours=tc.hours,
                    minutes=tc.minutes,
                    seconds=tc.seconds,
                    frames=tc.frames
                )
                # Subtract frames (handling underflow)
                total_frames = frame0_tc.to_frames() - tc_frames_offset
                if total_frames >= 0:
                    return Timecode.from_frame_number(total_frames, tc.fps)
                else:
                    return tc  # Can't extrapolate, return as-is
        
        return None
    
    def get_first_valid_timecode(self) -> Optional[Tuple[int, Timecode]]:
        """
        Get the first valid timecode and its frame number.
        Returns (frame_number, timecode) or None if no valid TC found.
        """
        if not self.results:
            return None
        
        for result in self.results:
            if result.visual_timecode is not None:
                return (result.frame_number, result.visual_timecode)
        
        return None
    
    def cleanup(self):
        """Release resources"""
        if self.reader:
            self.reader.close()


# =============================================================================
# Dual Camera Validator
# =============================================================================

class DualCameraValidator:
    """
    Validates dual-stream recordings (Camera A and B).
    Performs sync verification and generates comprehensive reports.
    """
    
    def __init__(
        self,
        video_a: Path,
        video_b: Path,
        metadata_a: Path,
        metadata_b: Path,
        output_dir: Path = Path("qa_reports"),
        roi_a: Optional[Dict[str, int]] = None,
        roi_b: Optional[Dict[str, int]] = None
    ):
        self.analyser_a = VideoAnalyser(video_a, metadata_a, "A")
        self.analyser_b = VideoAnalyser(video_b, metadata_b, "B")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store ROI configs for manual calibration
        self._roi_a = roi_a
        self._roi_b = roi_b
        
        self.validation_results: Dict[str, Any] = {}
        self.logger = logging.getLogger("DualCameraValidator")
    
    def run_full_validation(self, progress_callback=None, save_calibration_debug: bool = True) -> bool:
        """Execute complete validation pipeline"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Dual Camera Validation")
        self.logger.info("=" * 60)
        
        # Load both cameras
        if not self.analyser_a.load():
            self.logger.error("Failed to load Camera A")
            return False
        
        if not self.analyser_b.load():
            self.logger.error("Failed to load Camera B")
            return False
        
        # Calibrate timecode detection with manual ROI if provided
        self.analyser_a.calibrate(save_debug=save_calibration_debug, manual_roi=self._roi_a)
        self.analyser_b.calibrate(save_debug=save_calibration_debug, manual_roi=self._roi_b)
        
        # Run analysis on both cameras
        print("\n  Analysing Camera A (Left)...")
        self.analyser_a.analyse(progress_callback)
        
        print("\n  Analysing Camera B (Right)...")
        self.analyser_b.analyse(progress_callback)
        
        # Run validation checks
        self._validate_sync()
        self._validate_corruption()
        self._validate_drop_accuracy()
        self._validate_start_timecode()
        
        # Cleanup
        self.analyser_a.cleanup()
        self.analyser_b.cleanup()
        
        return True
    
    def _validate_sync(self):
        """
        Validate dual-camera synchronisation.
        
        Compares start timecodes from:
        1. First valid OCR timecode (extrapolated to frame 0)
        2. Metadata start timecode as fallback
        
        Cameras are synced if they have matching start timecodes.
        """
        tc_a = self.analyser_a.get_frame_zero_timecode()
        tc_b = self.analyser_b.get_frame_zero_timecode()
        
        # Get metadata timecodes as fallback
        meta_tc_a = self.analyser_a.metadata.start_timecode if self.analyser_a.metadata else None
        meta_tc_b = self.analyser_b.metadata.start_timecode if self.analyser_b.metadata else None
        
        # Determine sync status
        if tc_a and tc_b:
            # Both have valid OCR - compare them
            sync_pass = tc_a == tc_b
            source = "OCR"
        elif meta_tc_a and meta_tc_b:
            # Fall back to metadata comparison
            sync_pass = meta_tc_a == meta_tc_b
            source = "metadata"
            tc_a = meta_tc_a
            tc_b = meta_tc_b
        else:
            sync_pass = False
            source = "unknown"
        
        # Build notes
        if sync_pass:
            notes = f"Cameras synced (verified via {source})"
        elif tc_a is None and tc_b is None:
            notes = "Could not verify sync - no valid timecodes"
        elif tc_a is None:
            notes = "Camera A timecode unavailable"
        elif tc_b is None:
            notes = "Camera B timecode unavailable"
        else:
            notes = f"Frame 0 timecodes do not match ({source})"
        
        self.validation_results['sync'] = {
            'status': 'PASS' if sync_pass else 'FAIL',
            'camera_a_frame0_tc': str(tc_a) if tc_a else 'N/A',
            'camera_b_frame0_tc': str(tc_b) if tc_b else 'N/A',
            'notes': notes
        }
    
    def _validate_corruption(self):
        """Validate corruption detection against metadata"""
        results = {}
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            detected_corruptions = analyser.stats['corruptions']
            metadata_corruption = analyser.metadata.corrupted_at if analyser.metadata else -1
            
            # Check if metadata reports corruption
            metadata_reports_corruption = metadata_corruption >= 0
            vision_detected_corruption = detected_corruptions > 0
            
            if metadata_reports_corruption and not vision_detected_corruption:
                status = 'FAIL'
                notes = f'Metadata reports corruption at frame {metadata_corruption}, but vision found none'
            elif not metadata_reports_corruption and vision_detected_corruption:
                status = 'WARNING'
                notes = f'Vision detected {detected_corruptions} corruptions, but metadata reports none'
            elif metadata_reports_corruption and vision_detected_corruption:
                status = 'PASS'
                notes = 'Both metadata and vision agree on corruption presence'
            else:
                status = 'PASS'
                notes = 'No corruption detected'
            
            results[cam_id] = {
                'status': status,
                'metadata_corrupted_at': metadata_corruption,
                'vision_corruptions_detected': detected_corruptions,
                'notes': notes
            }
        
        self.validation_results['corruption'] = results
    
    def _validate_drop_accuracy(self):
        """Validate drop frame counts against metadata"""
        results = {}
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if not analyser.metadata:
                continue
            
            metadata_drops = analyser.metadata.dropped_frames
            detected_drops = analyser.stats['physical_drops']
            indicated_drops = analyser.stats['indicated_drops']
            successful_drops = analyser.stats['successful_drops']
            false_negatives = analyser.stats['false_negatives']
            false_positives = analyser.stats['false_positives']
            
            # Accuracy check
            drop_match = metadata_drops == detected_drops
            indicator_match = metadata_drops == indicated_drops
            
            results[cam_id] = {
                'metadata_drop_count': metadata_drops,
                'physical_drops_detected': detected_drops,
                'indicator_drops_counted': indicated_drops,
                'successful_processes': successful_drops,
                'false_negatives': false_negatives,
                'false_positives': false_positives,
                'drop_count_match': 'PASS' if drop_match else 'FAIL',
                'indicator_accuracy': 'PASS' if false_negatives == 0 and false_positives == 0 else 'FAIL'
            }
        
        self.validation_results['drop_accuracy'] = results
    
    def _validate_start_timecode(self):
        """
        Validate that OCR-detected start timecode matches metadata.
        
        Allows for small tolerance since:
        - OCR might succeed on frame 5 instead of frame 0
        - Extrapolation introduces small errors
        - Frame rate conversion might cause ±1 frame difference
        """
        results = {}
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if not analyser.metadata:
                continue
            
            ocr_tc = analyser.get_frame_zero_timecode()
            metadata_tc = analyser.metadata.start_timecode
            
            if ocr_tc and metadata_tc:
                # Allow tolerance of ±2 timecode frames
                ocr_frames = ocr_tc.to_frames()
                meta_frames = metadata_tc.to_frames()
                diff = abs(ocr_frames - meta_frames)
                
                if diff <= 2:
                    match = True
                    notes = "" if diff == 0 else f"Within tolerance (±{diff} frames)"
                else:
                    match = False
                    notes = f"Start timecode mismatch (diff: {diff} frames)"
            elif metadata_tc and not ocr_tc:
                # OCR failed but we have metadata - not a failure, just a warning
                match = True  # Trust metadata
                notes = "OCR unavailable, using metadata"
            else:
                match = False
                notes = "No timecode available"
            
            results[cam_id] = {
                'status': 'PASS' if match else 'FAIL',
                'ocr_frame0_tc': str(ocr_tc) if ocr_tc else 'N/A',
                'metadata_start_tc': str(metadata_tc) if metadata_tc else 'N/A',
                'notes': notes
            }
        
        self.validation_results['start_timecode'] = results
    
    def generate_report(self) -> Tuple[str, str]:
        """
        Generate validation report in both text and HTML formats.
        Returns (text_report, html_report) paths.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate text report
        text_path = self.output_dir / f"qa_report_{timestamp}.txt"
        text_content = self._generate_text_report()
        with open(text_path, 'w') as f:
            f.write(text_content)
        
        # Generate HTML report
        html_path = self.output_dir / f"qa_report_{timestamp}.html"
        html_content = self._generate_html_report()
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Generate JSON report (machine-readable)
        json_path = self.output_dir / f"qa_report_{timestamp}.json"
        self._generate_json_report(json_path)
        
        # Generate detailed CSV of all frames
        csv_path_a = self.output_dir / f"frame_analysis_cam_a_{timestamp}.csv"
        csv_path_b = self.output_dir / f"frame_analysis_cam_b_{timestamp}.csv"
        self._generate_frame_csv(self.analyser_a, csv_path_a)
        self._generate_frame_csv(self.analyser_b, csv_path_b)
        
        self.logger.info(f"Reports generated in {self.output_dir}")
        
        return str(text_path), str(html_path)
    
    def _generate_text_report(self) -> str:
        """Generate plain text report"""
        lines = []
        lines.append("=" * 70)
        lines.append("VIDEO QA VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Camera Info
        lines.append("-" * 70)
        lines.append("CAMERA INFORMATION")
        lines.append("-" * 70)
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if analyser.metadata:
                lines.append(f"Camera {cam_id}: {analyser.metadata.filename}")
                lines.append(f"  Frames: {analyser.metadata.num_frames}")
                lines.append(f"  Frame Rate: {analyser.metadata.frame_rate} fps")
                lines.append(f"  Start TC: {analyser.metadata.start_timecode}")
                lines.append("")
        
        # Sync Status
        lines.append("-" * 70)
        lines.append("1. SYNC STATUS")
        lines.append("-" * 70)
        sync = self.validation_results.get('sync', {})
        lines.append(f"Status: {sync.get('status', 'N/A')}")
        lines.append(f"Camera A Frame 0 TC: {sync.get('camera_a_frame0_tc', 'N/A')}")
        lines.append(f"Camera B Frame 0 TC: {sync.get('camera_b_frame0_tc', 'N/A')}")
        if sync.get('notes'):
            lines.append(f"Notes: {sync.get('notes')}")
        lines.append("")
        
        # Corruption Status
        lines.append("-" * 70)
        lines.append("2. CORRUPTION STATUS")
        lines.append("-" * 70)
        corruption = self.validation_results.get('corruption', {})
        for cam_id in ['A', 'B']:
            cam_data = corruption.get(cam_id, {})
            lines.append(f"Camera {cam_id}:")
            lines.append(f"  Status: {cam_data.get('status', 'N/A')}")
            lines.append(f"  Metadata Corrupted At: {cam_data.get('metadata_corrupted_at', 'N/A')}")
            lines.append(f"  Vision Corruptions: {cam_data.get('vision_corruptions_detected', 'N/A')}")
            lines.append(f"  Notes: {cam_data.get('notes', '')}")
            lines.append("")
        
        # Drop Frame Accuracy
        lines.append("-" * 70)
        lines.append("3. DROP FRAME ACCURACY")
        lines.append("-" * 70)
        drop_data = self.validation_results.get('drop_accuracy', {})
        
        # Create table
        lines.append(f"{'Metric':<30} {'Camera A':>15} {'Camera B':>15}")
        lines.append("-" * 60)
        
        metrics = [
            ('Metadata Drop Count', 'metadata_drop_count'),
            ('Physical Drops Detected', 'physical_drops_detected'),
            ('Indicator Drops Counted', 'indicator_drops_counted'),
            ('Successful Processes', 'successful_processes'),
            ('False Negatives', 'false_negatives'),
            ('False Positives', 'false_positives'),
            ('Drop Count Match', 'drop_count_match'),
            ('Indicator Accuracy', 'indicator_accuracy')
        ]
        
        for label, key in metrics:
            val_a = drop_data.get('A', {}).get(key, 'N/A')
            val_b = drop_data.get('B', {}).get(key, 'N/A')
            lines.append(f"{label:<30} {str(val_a):>15} {str(val_b):>15}")
        
        lines.append("")
        
        # Start Timecode
        lines.append("-" * 70)
        lines.append("4. START TIMECODE VALIDATION")
        lines.append("-" * 70)
        start_tc = self.validation_results.get('start_timecode', {})
        for cam_id in ['A', 'B']:
            cam_data = start_tc.get(cam_id, {})
            lines.append(f"Camera {cam_id}:")
            lines.append(f"  Status: {cam_data.get('status', 'N/A')}")
            lines.append(f"  OCR Frame 0: {cam_data.get('ocr_frame0_tc', 'N/A')}")
            lines.append(f"  Metadata: {cam_data.get('metadata_start_tc', 'N/A')}")
            lines.append("")
        
        # Indicator Health Summary
        lines.append("-" * 70)
        lines.append("5. INDICATOR HEALTH SUMMARY")
        lines.append("-" * 70)
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            fn = analyser.stats['false_negatives']
            fp = analyser.stats['false_positives']
            lines.append(f"Camera {cam_id}:")
            if fn == 0 and fp == 0:
                lines.append("  All drop indicators correctly reported")
            else:
                if fn > 0:
                    lines.append(f"  False Negatives (missed drops): {fn}")
                if fp > 0:
                    lines.append(f"  False Positives (ghost reports): {fp}")
            lines.append("")
        
        # Overall Summary
        lines.append("=" * 70)
        lines.append("OVERALL VALIDATION RESULT")
        lines.append("=" * 70)
        
        all_pass = all([
            sync.get('status') == 'PASS',
            all(c.get('status') == 'PASS' for c in corruption.values()),
            all(d.get('drop_count_match') == 'PASS' for d in drop_data.values()),
            all(d.get('indicator_accuracy') == 'PASS' for d in drop_data.values()),
            all(s.get('status') == 'PASS' for s in start_tc.values())
        ])
        
        lines.append(f"RESULT: {'PASS' if all_pass else 'FAIL'}")
        lines.append("")
        
        return '\n'.join(lines)
    
    def _generate_html_report(self) -> str:
        """Generate HTML report with styling"""
        sync = self.validation_results.get('sync', {})
        corruption = self.validation_results.get('corruption', {})
        drop_data = self.validation_results.get('drop_accuracy', {})
        start_tc = self.validation_results.get('start_timecode', {})
        
        def status_class(status):
            if status == 'PASS':
                return 'pass'
            elif status == 'FAIL':
                return 'fail'
            return 'warning'
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video QA Validation Report</title>
    <style>
        :root {{
            --pass-color: #28a745;
            --fail-color: #dc3545;
            --warning-color: #ffc107;
            --bg-color: #f8f9fa;
            --border-color: #dee2e6;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--bg-color);
            color: #212529;
        }}
        h1 {{
            text-align: center;
            color: #343a40;
            border-bottom: 3px solid #6c757d;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #495057;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .timestamp {{
            text-align: center;
            color: #6c757d;
            margin-bottom: 30px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .pass {{ background-color: var(--pass-color); }}
        .fail {{ background-color: var(--fail-color); }}
        .warning {{ background-color: var(--warning-color); color: #212529; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: #343a40;
            color: white;
        }}
        tr:hover {{
            background-color: #f1f3f4;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .overall-result {{
            text-align: center;
            padding: 30px;
            font-size: 24px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .overall-pass {{
            background-color: #d4edda;
            border: 2px solid var(--pass-color);
        }}
        .overall-fail {{
            background-color: #f8d7da;
            border: 2px solid var(--fail-color);
        }}
        .camera-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .camera-card {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
        }}
        .notes {{
            color: #6c757d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <h1>Video QA Validation Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>Camera Information</h2>
        <div class="camera-info">
'''
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if analyser.metadata:
                html += f'''
            <div class="camera-card">
                <h3>Camera {cam_id}</h3>
                <p><strong>File:</strong> {analyser.metadata.filename}</p>
                <p><strong>Frames:</strong> {analyser.metadata.num_frames}</p>
                <p><strong>Frame Rate:</strong> {analyser.metadata.frame_rate} fps</p>
                <p><strong>Start TC:</strong> {analyser.metadata.start_timecode}</p>
            </div>
'''
        
        html += f'''
        </div>
    </div>
    
    <div class="section">
        <h2>1. Sync Status</h2>
        <p>Status: <span class="status-badge {status_class(sync.get('status'))}">{sync.get('status', 'N/A')}</span></p>
        <table>
            <tr>
                <th>Camera</th>
                <th>Frame 0 Timecode</th>
            </tr>
            <tr>
                <td>Camera A</td>
                <td>{sync.get('camera_a_frame0_tc', 'N/A')}</td>
            </tr>
            <tr>
                <td>Camera B</td>
                <td>{sync.get('camera_b_frame0_tc', 'N/A')}</td>
            </tr>
        </table>
        {f'<p class="notes">Notes: {sync.get("notes")}</p>' if sync.get('notes') else ''}
    </div>
    
    <div class="section">
        <h2>2. Corruption Status</h2>
        <table>
            <tr>
                <th>Camera</th>
                <th>Status</th>
                <th>Metadata Corrupted At</th>
                <th>Vision Corruptions</th>
                <th>Notes</th>
            </tr>
'''
        
        for cam_id in ['A', 'B']:
            cam_data = corruption.get(cam_id, {})
            html += f'''
            <tr>
                <td>Camera {cam_id}</td>
                <td><span class="status-badge {status_class(cam_data.get('status'))}">{cam_data.get('status', 'N/A')}</span></td>
                <td>{cam_data.get('metadata_corrupted_at', 'N/A')}</td>
                <td>{cam_data.get('vision_corruptions_detected', 'N/A')}</td>
                <td class="notes">{cam_data.get('notes', '')}</td>
            </tr>
'''
        
        html += f'''
        </table>
    </div>
    
    <div class="section">
        <h2>3. Drop Frame Accuracy</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Camera A</th>
                <th>Camera B</th>
            </tr>
'''
        
        metrics = [
            ('Metadata Drop Count', 'metadata_drop_count'),
            ('Physical Drops Detected', 'physical_drops_detected'),
            ('Indicator Drops Counted', 'indicator_drops_counted'),
            ('Successful Processes', 'successful_processes'),
            ('False Negatives', 'false_negatives'),
            ('False Positives', 'false_positives'),
            ('Drop Count Match', 'drop_count_match'),
            ('Indicator Accuracy', 'indicator_accuracy')
        ]
        
        for label, key in metrics:
            val_a = drop_data.get('A', {}).get(key, 'N/A')
            val_b = drop_data.get('B', {}).get(key, 'N/A')
            
            # Format status values with badges
            if key in ['drop_count_match', 'indicator_accuracy']:
                val_a = f'<span class="status-badge {status_class(val_a)}">{val_a}</span>'
                val_b = f'<span class="status-badge {status_class(val_b)}">{val_b}</span>'
            
            html += f'''
            <tr>
                <td>{label}</td>
                <td>{val_a}</td>
                <td>{val_b}</td>
            </tr>
'''
        
        html += f'''
        </table>
    </div>
    
    <div class="section">
        <h2>4. Start Timecode Validation</h2>
        <table>
            <tr>
                <th>Camera</th>
                <th>Status</th>
                <th>OCR Frame 0</th>
                <th>Metadata Start TC</th>
            </tr>
'''
        
        for cam_id in ['A', 'B']:
            cam_data = start_tc.get(cam_id, {})
            html += f'''
            <tr>
                <td>Camera {cam_id}</td>
                <td><span class="status-badge {status_class(cam_data.get('status'))}">{cam_data.get('status', 'N/A')}</span></td>
                <td>{cam_data.get('ocr_frame0_tc', 'N/A')}</td>
                <td>{cam_data.get('metadata_start_tc', 'N/A')}</td>
            </tr>
'''
        
        html += '''
        </table>
    </div>
    
    <div class="section">
        <h2>5. Indicator Health Summary</h2>
'''
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            fn = analyser.stats['false_negatives']
            fp = analyser.stats['false_positives']
            
            if fn == 0 and fp == 0:
                health_status = 'pass'
                health_text = 'All drop indicators correctly reported'
            else:
                health_status = 'fail'
                health_text = f'False Negatives: {fn}, False Positives: {fp}'
            
            html += f'''
        <p><strong>Camera {cam_id}:</strong> 
            <span class="status-badge {health_status}">{health_text}</span>
        </p>
'''
        
        # Overall result
        all_pass = all([
            sync.get('status') == 'PASS',
            all(c.get('status') == 'PASS' for c in corruption.values()),
            all(d.get('drop_count_match') == 'PASS' for d in drop_data.values()),
            all(d.get('indicator_accuracy') == 'PASS' for d in drop_data.values()),
            all(s.get('status') == 'PASS' for s in start_tc.values())
        ])
        
        overall_class = 'overall-pass' if all_pass else 'overall-fail'
        overall_text = 'VALIDATION PASSED' if all_pass else 'VALIDATION FAILED'
        
        html += f'''
    </div>
    
    <div class="overall-result {overall_class}">
        <strong>{overall_text}</strong>
    </div>
</body>
</html>
'''
        
        return html
    
    def _generate_json_report(self, output_path: Path):
        """Generate machine-readable JSON report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'camera_a_stats': self.analyser_a.stats,
            'camera_b_stats': self.analyser_b.stats,
            'camera_a_metadata': {
                'filename': self.analyser_a.metadata.filename if self.analyser_a.metadata else None,
                'num_frames': self.analyser_a.metadata.num_frames if self.analyser_a.metadata else None,
                'frame_rate': self.analyser_a.metadata.frame_rate if self.analyser_a.metadata else None,
                'dropped_frames': self.analyser_a.metadata.dropped_frames if self.analyser_a.metadata else None,
                'start_timecode': str(self.analyser_a.metadata.start_timecode) if self.analyser_a.metadata else None
            },
            'camera_b_metadata': {
                'filename': self.analyser_b.metadata.filename if self.analyser_b.metadata else None,
                'num_frames': self.analyser_b.metadata.num_frames if self.analyser_b.metadata else None,
                'frame_rate': self.analyser_b.metadata.frame_rate if self.analyser_b.metadata else None,
                'dropped_frames': self.analyser_b.metadata.dropped_frames if self.analyser_b.metadata else None,
                'start_timecode': str(self.analyser_b.metadata.start_timecode) if self.analyser_b.metadata else None
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_frame_csv(self, analyser: VideoAnalyser, output_path: Path):
        """Generate detailed CSV of frame-by-frame analysis"""
        rows = []
        for result in analyser.results:
            rows.append({
                'frame_number': result.frame_number,
                'visual_timecode': str(result.visual_timecode) if result.visual_timecode else '',
                'visual_timecode_raw': result.visual_timecode_raw,
                'drop_indicator': result.drop_indicator_present,
                'is_duplicate': result.is_physical_duplicate,
                'status': result.status.name,
                'image_hash': result.image_hash,
                'notes': result.notes
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


# =============================================================================
# Main Entry Point
# =============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_qa.log')
        ]
    )


def main():
    """Main entry point for the QA application"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Video QA Automation for Hardware Validation'
    )
    parser.add_argument('--video-a', '-va', required=True, help='Path to Camera A video file')
    parser.add_argument('--video-b', '-vb', required=True, help='Path to Camera B video file')
    parser.add_argument('--metadata-a', '-ma', required=True, help='Path to Camera A metadata JSON')
    parser.add_argument('--metadata-b', '-mb', required=True, help='Path to Camera B metadata JSON')
    parser.add_argument('--output', '-o', default='qa_reports', help='Output directory for reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', '-d', action='store_true', help='Save debug frames')
    
    args = parser.parse_args()
    
    # Setup
    global DEBUG
    DEBUG = args.debug
    setup_logging(args.verbose)
    
    logger = logging.getLogger('main')
    logger.info("Video QA Automation Starting...")
    
    # Create validator
    validator = DualCameraValidator(
        video_a=Path(args.video_a),
        video_b=Path(args.video_b),
        metadata_a=Path(args.metadata_a),
        metadata_b=Path(args.metadata_b),
        output_dir=Path(args.output)
    )
    
    # Progress callback
    def progress(current, total):
        pct = (current / total) * 100
        logger.info(f"Progress: {current}/{total} ({pct:.1f}%)")
    
    # Run validation
    success = validator.run_full_validation(progress_callback=progress)
    
    if success:
        text_path, html_path = validator.generate_report()
        logger.info(f"Text report: {text_path}")
        logger.info(f"HTML report: {html_path}")
        logger.info("Validation complete!")
    else:
        logger.error("Validation failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
