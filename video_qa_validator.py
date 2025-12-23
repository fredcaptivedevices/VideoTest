#!/usr/bin/env python3
"""
Video QA Automation Application for Hardware Validation
========================================================
Processes dual-stream video (Camera A/B) and metadata to validate recording integrity.

Author: Captive Devices QA System
Version: 2.0.0 - Dramatically improved OCR pipeline
"""

import cv2
import numpy as np
import json
import re
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime
from enum import Enum, auto

# Try to import OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available, using EasyOCR only")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available, using Tesseract only")


# =============================================================================
# Configuration & Constants
# =============================================================================

DEBUG = True  # Set to True to save screenshots of flagged frames
DEBUG_OUTPUT_DIR = Path("debug_frames")

# Debug: Save OCR input images for troubleshooting
DEBUG_OCR = False  # Set to True to save OCR debug images (SLOW - disk I/O)
DEBUG_OCR_DIR = Path("debug_ocr")
DEBUG_OCR_SAVE_ALL = False  # Set True to save ALL frames, False for first N only
DEBUG_OCR_FIRST_N = 20  # Save debug images for first N frames

# EasyOCR reader - initialised lazily for performance
_EASYOCR_READER = None

def get_ocr_reader():
    """
    Lazy initialisation of EasyOCR reader (only created once).
    Automatically uses GPU if available.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None and EASYOCR_AVAILABLE:
        try:
            import torch

            # Check for GPU availability
            use_gpu = False
            gpu_name = "None"

            if torch.cuda.is_available():
                use_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  [OCR] Using NVIDIA GPU: {gpu_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                use_gpu = True
                gpu_name = "Apple Silicon (MPS)"
                print(f"  [OCR] Using Apple Silicon GPU (Metal)")
            else:
                print("  [OCR] No GPU detected, using CPU")

            _EASYOCR_READER = easyocr.Reader(
                ['en'],
                gpu=use_gpu,
                verbose=False,
                model_storage_directory=None,
                download_enabled=True
            )
        except Exception as e:
            print(f"  [OCR] Failed to initialize EasyOCR: {e}")
            _EASYOCR_READER = None
    return _EASYOCR_READER


# Drop indicator ROI - Fixed position based on screenshot analysis
# The "Dropped frame" text appears in top-left corner
ROI_DROP = {
    'x': 0,
    'y': 0,
    'width': 300,
    'height': 60
}

# Timecode patterns for OCR validation
# Standard: HH:MM:SS:FF with colons
TIMECODE_PATTERN = re.compile(r'(\d{1,2}):(\d{2}):(\d{2}):(\d{2})')
# Loose: digits with any separator or no separator
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
DEFAULT_TIMECODE_FPS = 30  # LockitSlate default timecode display rate


def compute_frame_rate_multiplier(video_fps: int, timecode_fps: int = DEFAULT_TIMECODE_FPS) -> int:
    """
    Compute how many video frames share each timecode value.
    """
    if video_fps <= 0 or timecode_fps <= 0:
        return 1

    multiplier = video_fps // timecode_fps
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
            h, m, s, f = int(digits_only[0:2]), int(digits_only[2:4]), int(digits_only[4:6]), int(digits_only[6:8])
            if is_valid_timecode(h, m, s, f):
                return cls(hours=h, minutes=m, seconds=s, frames=f)
        elif len(digits_only) == 7:
            h, m, s, f = int(digits_only[0:1]), int(digits_only[1:3]), int(digits_only[3:5]), int(digits_only[5:7])
            if is_valid_timecode(h, m, s, f):
                return cls(hours=h, minutes=m, seconds=s, frames=f)
        elif len(digits_only) == 6:
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

    @classmethod
    def from_frame_number(cls, total_frames: int, fps: int = 60) -> 'Timecode':
        """Create Timecode from total frame count"""
        if total_frames < 0:
            total_frames = 0

        frames = total_frames % fps
        total_seconds = total_frames // fps
        seconds = total_seconds % 60
        total_minutes = total_seconds // 60
        minutes = total_minutes % 60
        hours = total_minutes // 60

        return cls(hours=hours, minutes=minutes, seconds=seconds, frames=frames)

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
    ocr_confidence: float = 0.0


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
# Enhanced OCR Pipeline for LED Dot-Matrix Displays
# =============================================================================

class LEDTimecodeOCR:
    """
    Specialized OCR pipeline for LED dot-matrix timecode displays.

    Key features:
    - Multiple preprocessing strategies optimized for LED displays
    - Dual OCR engine support (EasyOCR + Tesseract)
    - Confidence-based result selection
    - Robust timecode pattern matching
    """

    def __init__(self, camera_id: str = "A"):
        self.camera_id = camera_id
        self.logger = logging.getLogger(f"LEDTimecodeOCR_{camera_id}")
        self._frame_count = 0

        # Track OCR success rate for debugging
        self.stats = {
            'attempts': 0,
            'successes': 0,
            'tesseract_wins': 0,
            'easyocr_wins': 0
        }

        # Cache successful preprocessing methods for faster subsequent frames
        # Format: [(preprocess_name, scale, engine), ...]
        self._successful_methods: List[Tuple[str, float, str]] = []
        self._max_cached_methods = 5

    def preprocess_led_display(self, roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Generate multiple preprocessed versions of the LED display ROI.

        LED dot-matrix displays require special handling:
        1. Dots need to be fused into solid digit shapes
        2. Background must be cleanly separated from foreground
        3. Digit segments need to be connected properly

        Returns list of (name, preprocessed_image) tuples for OCR attempts.
        """
        if roi is None or roi.size == 0:
            return []

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        results = []

        # Check if this is a light-on-dark or dark-on-light display
        mean_val = np.mean(gray)
        is_light_on_dark = mean_val < 128

        # =====================================================================
        # Strategy 1: Heavy Gaussian blur to fuse LED dots
        # =====================================================================
        for blur_size in [9, 11, 15]:
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

            # Multiple threshold levels
            for thresh_val in [180, 160, 140, 200, 220]:
                _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

                # Apply morphological closing to connect digit segments
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

                # For OCR, we need BLACK text on WHITE background (standard convention)
                # After thresholding LED displays (light-on-dark): white digits on black
                # After thresholding documents (dark-on-light): depends on threshold
                # Always invert light-on-dark to get black text on white for OCR
                if is_light_on_dark:
                    closed = cv2.bitwise_not(closed)

                results.append((f'blur{blur_size}_t{thresh_val}', closed))

        # =====================================================================
        # Strategy 2: Otsu's threshold (auto-level detection)
        # =====================================================================
        for blur_size in [7, 11]:
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

            # For OCR: black text on white background
            if is_light_on_dark:
                closed = cv2.bitwise_not(closed)

            results.append((f'otsu_blur{blur_size}', closed))

        # =====================================================================
        # Strategy 3: Adaptive threshold (handles uneven lighting)
        # =====================================================================
        for blur_size in [7, 11]:
            blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

            # Adaptive threshold with different block sizes
            for block_size in [31, 51, 71]:
                adaptive = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size, -3
                )

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)

                # For OCR: black text on white background
                if is_light_on_dark:
                    closed = cv2.bitwise_not(closed)

                results.append((f'adaptive_b{blur_size}_bs{block_size}', closed))

        # =====================================================================
        # Strategy 4: Morphological reconstruction for dot-matrix
        # =====================================================================
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        for thresh_val in [160, 180, 200]:
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

            # Heavy dilation to connect dots, then erode to restore shape
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)

            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded = cv2.erode(dilated, kernel_erode, iterations=1)

            # For OCR: black text on white background
            if is_light_on_dark:
                eroded = cv2.bitwise_not(eroded)

            results.append((f'morph_t{thresh_val}', eroded))

        # =====================================================================
        # Strategy 5: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # =====================================================================
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)

        for thresh_val in [180, 200]:
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # For OCR: black text on white background
            if is_light_on_dark:
                closed = cv2.bitwise_not(closed)

            results.append((f'clahe_t{thresh_val}', closed))

        return results

    def ocr_with_tesseract(self, img: np.ndarray, scale: float = 2.0) -> Tuple[Optional[Timecode], str, float]:
        """
        Run Tesseract OCR on preprocessed image.

        Returns (timecode, raw_text, confidence)
        """
        if not TESSERACT_AVAILABLE:
            return None, "", 0.0

        try:
            # Scale up for better recognition
            if scale != 1.0:
                img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                img_scaled = img

            # Tesseract config for digits only
            config = '--psm 7 -c tessedit_char_whitelist=0123456789:'

            # Get text with confidence data
            data = pytesseract.image_to_data(img_scaled, config=config, output_type=pytesseract.Output.DICT)

            # Combine text and calculate average confidence
            texts = []
            confidences = []
            for i, text in enumerate(data['text']):
                text = text.strip()
                conf = data['conf'][i]
                if text and conf > 0:
                    texts.append(text)
                    confidences.append(conf)

            raw_text = ''.join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            # Clean up and parse
            raw_text = self._clean_ocr_text(raw_text)
            tc = Timecode.from_string(raw_text)

            return tc, raw_text, avg_conf

        except Exception as e:
            self.logger.debug(f"Tesseract error: {e}")
            return None, "", 0.0

    def ocr_with_easyocr(self, img: np.ndarray, scale: float = 2.0) -> Tuple[Optional[Timecode], str, float]:
        """
        Run EasyOCR on preprocessed image with multiple configurations.

        Returns (timecode, raw_text, confidence)
        """
        reader = get_ocr_reader()
        if reader is None:
            return None, "", 0.0

        try:
            # Scale up for better recognition
            if scale != 1.0:
                img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                img_scaled = img

            # Convert to BGR if grayscale (EasyOCR expects this)
            if len(img_scaled.shape) == 2:
                img_for_ocr = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
            else:
                img_for_ocr = img_scaled

            # Multiple EasyOCR configurations to try (optimized for LED displays)
            easyocr_configs = [
                # Config 1: Standard with low thresholds
                {
                    'allowlist': '0123456789:',
                    'paragraph': False,
                    'detail': 1,
                    'width_ths': 0.5,
                    'height_ths': 0.5,
                    'contrast_ths': 0.1,
                    'adjust_contrast': 0.5
                },
                # Config 2: Lower text threshold for faint digits
                {
                    'allowlist': '0123456789:',
                    'paragraph': False,
                    'detail': 1,
                    'text_threshold': 0.5,
                    'low_text': 0.3,
                    'link_threshold': 0.3,
                },
                # Config 3: Very permissive
                {
                    'allowlist': '0123456789:',
                    'paragraph': False,
                    'detail': 1,
                    'width_ths': 0.3,
                    'height_ths': 0.3,
                    'text_threshold': 0.4,
                    'low_text': 0.2,
                },
                # Config 4: Paragraph mode (better for connected text)
                {
                    'allowlist': '0123456789:',
                    'paragraph': True,
                    'detail': 1,
                    'width_ths': 0.7,
                },
                # Config 5: High magnification for small text
                {
                    'allowlist': '0123456789:',
                    'paragraph': False,
                    'detail': 1,
                    'mag_ratio': 2.0,
                    'text_threshold': 0.6,
                },
            ]

            best_tc = None
            best_raw = ""
            best_conf = 0.0

            for config in easyocr_configs:
                try:
                    results = reader.readtext(img_for_ocr, **config)

                    if not results:
                        continue

                    # Sort by x-position and combine
                    sorted_results = sorted(results, key=lambda r: r[0][0][0])

                    texts = []
                    confidences = []
                    for (bbox, text, conf) in sorted_results:
                        text = text.strip()
                        if text:
                            texts.append(text)
                            confidences.append(conf)

                    if not texts:
                        continue

                    raw_text = ''.join(texts)
                    avg_conf = sum(confidences) / len(confidences) * 100

                    # Clean and try to parse
                    cleaned = self._clean_ocr_text(raw_text)
                    tc = Timecode.from_string(cleaned)

                    # If valid timecode and better confidence, keep it
                    if tc and avg_conf > best_conf:
                        best_tc = tc
                        best_raw = cleaned
                        best_conf = avg_conf

                    # Early exit on high confidence valid result
                    if best_tc and best_conf > 85:
                        break

                except Exception:
                    continue

            return best_tc, best_raw, best_conf

        except Exception as e:
            self.logger.debug(f"EasyOCR error: {e}")
            return None, "", 0.0

    def _clean_ocr_text(self, text: str) -> str:
        """Clean up common OCR errors for timecode recognition"""
        if not text:
            return ""

        # Replace common OCR misreads
        replacements = {
            '.': ':',   # Period to colon (common separator confusion)
            ',': ':',   # Comma to colon
            ';': ':',   # Semicolon to colon
            ' ': '',    # Remove spaces
            'O': '0',   # O to zero
            'o': '0',   # lowercase o to zero
            'Q': '0',   # Q to zero
            'D': '0',   # D to zero (closed loop)
            'l': '1',   # lowercase L to one
            'I': '1',   # uppercase I to one
            '|': '1',   # pipe to one
            'i': '1',   # lowercase i to one
            '!': '1',   # exclamation to one
            'S': '5',   # S to five
            's': '5',   # lowercase s to five
            'B': '8',   # B to eight
            'Z': '2',   # Z to two
            'z': '2',   # lowercase z to two
            'G': '6',   # G to six
            'g': '9',   # g to nine (loop at bottom)
            'q': '9',   # q to nine
            'A': '4',   # A to four
            'b': '6',   # b to six
            'T': '7',   # T to seven
            '\n': '',   # Remove newlines
            '\r': '',   # Remove carriage returns
            '-': ':',   # Hyphen to colon (sometimes used as separator)
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def extract_timecode(self, roi: np.ndarray) -> Tuple[Optional[Timecode], str, float]:
        """
        Extract timecode from ROI using multiple preprocessing strategies and OCR engines.

        Uses method caching to prioritize previously successful strategies.

        Returns (timecode, raw_text, confidence)
        """
        self.stats['attempts'] += 1
        self._frame_count += 1

        if roi is None or roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 50:
            return None, "", 0.0

        # Generate preprocessed images
        preprocessed_images = self.preprocess_led_display(roi)
        preprocessed_dict = {name: img for name, img in preprocessed_images}

        # Debug: Save preprocessing results for first N frames
        if DEBUG_OCR and (DEBUG_OCR_SAVE_ALL or self._frame_count <= DEBUG_OCR_FIRST_N):
            self._save_debug_images(roi, preprocessed_images)

        # Track best result
        best_tc = None
        best_raw = ""
        best_conf = 0.0
        best_method = ""

        # PHASE 1: Try cached successful methods first (much faster)
        for cached_name, cached_scale, cached_engine in self._successful_methods:
            if cached_name in preprocessed_dict:
                img = preprocessed_dict[cached_name]

                if cached_engine == 'tesseract' and TESSERACT_AVAILABLE:
                    tc, raw, conf = self.ocr_with_tesseract(img, cached_scale)
                elif cached_engine == 'easyocr' and EASYOCR_AVAILABLE:
                    tc, raw, conf = self.ocr_with_easyocr(img, cached_scale)
                else:
                    continue

                if tc and conf > best_conf:
                    best_tc = tc
                    best_raw = raw
                    best_conf = conf
                    best_method = f"{cached_engine}_{cached_name}_s{cached_scale}"

                # If cached method works well, use it
                if best_tc and best_conf > 70:
                    break

        # PHASE 2: Full search if cached methods didn't work well
        if not best_tc or best_conf < 70:
            for name, img in preprocessed_images:
                # Skip methods we already tried from cache
                already_tried = any(
                    name == cn for cn, _, _ in self._successful_methods
                )

                for scale in [2.0, 3.0, 2.5]:
                    # Try Tesseract
                    if TESSERACT_AVAILABLE:
                        tc, raw, conf = self.ocr_with_tesseract(img, scale)
                        if tc and conf > best_conf:
                            best_tc = tc
                            best_raw = raw
                            best_conf = conf
                            best_method = f"tesseract_{name}_s{scale}"

                    # Try EasyOCR
                    if EASYOCR_AVAILABLE:
                        tc, raw, conf = self.ocr_with_easyocr(img, scale)
                        if tc and conf > best_conf:
                            best_tc = tc
                            best_raw = raw
                            best_conf = conf
                            best_method = f"easyocr_{name}_s{scale}"

                # Early exit if we get a high-confidence result
                if best_conf > 80:
                    break

        # Update cache with successful method
        if best_tc and best_method:
            # Parse method string to extract components
            parts = best_method.split('_')
            if len(parts) >= 3:
                engine = parts[0]
                # Reconstruct preprocess name (everything between engine and scale)
                scale_part = parts[-1]  # e.g., "s2.0"
                scale = float(scale_part[1:]) if scale_part.startswith('s') else 2.0
                preprocess_name = '_'.join(parts[1:-1])

                # Add to cache if not already there
                cache_entry = (preprocess_name, scale, engine)
                if cache_entry not in self._successful_methods:
                    self._successful_methods.insert(0, cache_entry)
                    # Keep cache limited
                    if len(self._successful_methods) > self._max_cached_methods:
                        self._successful_methods.pop()

        # Update stats
        if best_tc:
            self.stats['successes'] += 1
            if 'tesseract' in best_method:
                self.stats['tesseract_wins'] += 1
            else:
                self.stats['easyocr_wins'] += 1

        return best_tc, best_raw, best_conf

    def _save_debug_images(self, roi: np.ndarray, preprocessed: List[Tuple[str, np.ndarray]]):
        """Save debug images for troubleshooting"""
        try:
            DEBUG_OCR_DIR.mkdir(exist_ok=True)

            # Save original ROI
            cv2.imwrite(
                str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._frame_count:04d}_00_roi.png"),
                roi
            )

            # Save first few preprocessed images
            for i, (name, img) in enumerate(preprocessed[:8]):
                cv2.imwrite(
                    str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._frame_count:04d}_{i+1:02d}_{name}.png"),
                    img
                )
        except Exception as e:
            self.logger.debug(f"Failed to save debug images: {e}")


# =============================================================================
# Frame Reader Class
# =============================================================================

class FrameReader:
    """
    Handles frame extraction and analysis from video files.
    Includes drop zone detection and timecode OCR.
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
        self.frame_rate_multiplier = frame_rate_multiplier

        # ROI for timecode - can be auto-detected or manually set
        self.roi_timecode: Optional[Dict[str, int]] = None
        self.timecode_calibrated: bool = False

        # Enhanced OCR pipeline
        self.ocr_pipeline = LEDTimecodeOCR(camera_id)

        # Timecode-based duplicate detection
        self._prev_timecode: Optional['Timecode'] = None
        self._consecutive_same_tc: int = 0

        # Legacy image-based detection (fallback)
        self._prev_frame_small: Optional[np.ndarray] = None
        self._hash_size: int = 16
        self._consecutive_duplicates: int = 0

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
        """Read the next frame sequentially"""
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        return frame if ret else None

    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame from the video"""
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
        """
        # Extract drop indicator region
        roi = frame[
            ROI_DROP['y']:ROI_DROP['y'] + ROI_DROP['height'],
            ROI_DROP['x']:ROI_DROP['x'] + ROI_DROP['width']
        ]

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Quick check - if region is too dark, skip OCR
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return False

        # Check for white pixels (text) in the region
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white_pixel_ratio = np.sum(thresh > 0) / thresh.size

        # If less than 1% white pixels, probably no text
        if white_pixel_ratio < 0.01:
            return False

        # Use OCR to detect "Dropped" or "DROP" text
        detected = False

        # Try Tesseract first (faster for this use case)
        if TESSERACT_AVAILABLE:
            try:
                ocr_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
                text = pytesseract.image_to_string(thresh, config=ocr_config).strip().lower()
                drop_keywords = ['drop', 'dropped', 'frame']
                detected = any(keyword in text for keyword in drop_keywords)
            except Exception:
                pass

        # Fallback to EasyOCR if needed
        if not detected and EASYOCR_AVAILABLE:
            try:
                reader = get_ocr_reader()
                if reader:
                    results = reader.readtext(thresh, detail=0)
                    text = ' '.join(results).lower()
                    drop_keywords = ['drop', 'dropped', 'frame']
                    detected = any(keyword in text for keyword in drop_keywords)
            except Exception:
                pass

        return detected

    def auto_detect_timecode_roi(self, frame: np.ndarray) -> bool:
        """
        Auto-detect the timecode region on the first frame.
        Enhanced detection for LED dot-matrix displays.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        self.logger.info(f"Auto-detecting timecode ROI in {width}x{height} frame...")

        # Strategy 1: Look for bright LED regions
        for thresh_val in [220, 200, 180, 160]:
            _, bright_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            # Dilate to connect LED segments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
            dilated = cv2.dilate(bright_mask, kernel, iterations=3)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours[:20]:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h

                # Timecode display: wide rectangle (aspect ratio 3:1 to 8:1)
                if 2.5 < aspect_ratio < 10.0 and w > 80 and h > 15 and area > 2000:
                    # Add generous padding
                    padding_x = 60
                    padding_y = 40
                    roi_x = max(0, x - padding_x)
                    roi_y = max(0, y - padding_y)
                    roi_w = min(width - roi_x, w + 2 * padding_x)
                    roi_h = min(height - roi_y, h + 2 * padding_y)

                    # Test OCR on this region
                    test_roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    tc, _, conf = self.ocr_pipeline.extract_timecode(test_roi)

                    if tc:
                        self.roi_timecode = {'x': roi_x, 'y': roi_y, 'width': roi_w, 'height': roi_h}
                        self.timecode_calibrated = True
                        self.logger.info(f"Timecode ROI detected (thresh={thresh_val}): x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                        self.logger.info(f"  Initial timecode: {tc} (confidence: {conf:.1f})")
                        return True

        # Strategy 2: Grid-based scan
        self.logger.info("Bright region detection failed, trying grid scan...")

        window_sizes = [(500, 120), (450, 100), (400, 90), (350, 80)]

        for win_w, win_h in window_sizes:
            step_x = win_w // 4
            step_y = win_h // 3

            for y in range(0, height - win_h, step_y):
                for x in range(0, width - win_w, step_x):
                    test_roi = gray[y:y+win_h, x:x+win_w]

                    # Skip very dark regions
                    if np.max(test_roi) < 100:
                        continue

                    tc, _, conf = self.ocr_pipeline.extract_timecode(test_roi)

                    if tc:
                        padding = 50
                        self.roi_timecode = {
                            'x': max(0, x - padding),
                            'y': max(0, y - padding),
                            'width': min(win_w + 2*padding, width - x + padding),
                            'height': min(win_h + 2*padding, height - y + padding)
                        }
                        self.timecode_calibrated = True
                        self.logger.info(f"Timecode ROI detected (grid scan): {self.roi_timecode}")
                        self.logger.info(f"  Initial timecode: {tc}")
                        return True

        # Fallback: Use large center-lower region
        self.logger.warning("Auto-detection failed, using default centre ROI")
        self.roi_timecode = {
            'x': 100,
            'y': height // 4,
            'width': width - 200,
            'height': height // 2
        }
        return False

    def extract_timecode(self, frame: np.ndarray) -> Tuple[Optional[Timecode], str]:
        """
        Extract timecode from frame using enhanced OCR pipeline.
        Returns tuple of (Timecode object, raw OCR text).

        OPTIMISED for speed and accuracy:
        - ROI-level caching to skip OCR on identical frames (major speedup)
        - Smart threshold selection based on LED brightness (avoids 5x OCR overhead)
        - CLAHE contrast enhancement for low-light displays
        - Caching of last successful threshold method
        - Early exit on successful OCR
        """
        if not self.roi_timecode:
            return None, ""

        # Extract ROI
        roi = frame[
            self.roi_timecode['y']:self.roi_timecode['y'] + self.roi_timecode['height'],
            self.roi_timecode['x']:self.roi_timecode['x'] + self.roi_timecode['width']
        ]

        # OPTIMIZATION: ROI-level caching to skip OCR on nearly identical frames
        # Compute a fast hash of the ROI to detect unchanged timecode regions
        if not hasattr(self, '_roi_cache'):
            self._roi_cache = {'hash': None, 'result': (None, "")}

        # Compute simple hash: mean + std of ROI (very fast)
        roi_hash = (int(np.mean(roi)), int(np.std(roi)))
        if roi_hash == self._roi_cache['hash']:
            # ROI is identical to previous frame - return cached result
            return self._roi_cache['result']

        # For monochrome footage, all channels are the same
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # OPTIMIZATION: Check if ROI is too dark (no LED visible) - skip OCR entirely
        mean_brightness = np.mean(gray)
        if mean_brightness < 10:
            result = (None, "")
            self._roi_cache = {'hash': roi_hash, 'result': result}
            return result

        # OPTIMIZATION: Apply CLAHE for better contrast on dim displays
        if mean_brightness < 80:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

        # Step 1: Apply Gaussian blur to fuse dot-matrix LED dots
        blur_kernel = 5  # Slightly smaller for speed, still effective
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Morphological kernel for closing operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Debug: Save first few ROI images (only when DEBUG_OCR is enabled)
        if DEBUG_OCR:
            if not hasattr(self, '_debug_ocr_count'):
                self._debug_ocr_count = 0
            self._debug_ocr_count += 1
            if self._debug_ocr_count <= 10:
                try:
                    DEBUG_OCR_DIR.mkdir(exist_ok=True)
                    cv2.imwrite(str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._debug_ocr_count:04d}_1_roi.png"), roi)
                    cv2.imwrite(str(DEBUG_OCR_DIR / f"cam{self.camera_id}_frame{self._debug_ocr_count:04d}_2_gray.png"), gray)
                except Exception as e:
                    self.logger.debug(f"Failed to save debug OCR image: {e}")

        # OPTIMIZATION: Smart threshold selection based on brightness
        # Instead of trying all 5 thresholds, select 1-2 based on image characteristics
        max_val = np.max(blurred)

        # Initialize cache for last successful threshold method
        if not hasattr(self, '_last_successful_thresh'):
            self._last_successful_thresh = None

        images_to_try = []

        # Detect if this is light-on-dark (LED display) for inversion
        # LED displays have white dots on dark background, so mean is low
        is_light_on_dark = mean_brightness < 128

        # If we have a cached successful method, try it first
        if self._last_successful_thresh:
            thresh_val = self._last_successful_thresh
            if thresh_val == 'otsu':
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # Invert LED displays to get black text on white (standard OCR format)
            if is_light_on_dark:
                thresh = cv2.bitwise_not(thresh)
            images_to_try.append((self._last_successful_thresh, thresh))

        # Smart threshold selection based on max brightness
        if max_val > 200:
            # Bright LED - use high threshold
            if self._last_successful_thresh != 200:
                _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                # Invert LED displays to get black text on white (standard OCR format)
                if is_light_on_dark:
                    thresh = cv2.bitwise_not(thresh)
                images_to_try.append((200, thresh))
        elif max_val > 150:
            # Medium brightness - use medium threshold
            if self._last_successful_thresh != 170:
                _, thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                # Invert LED displays to get black text on white (standard OCR format)
                if is_light_on_dark:
                    thresh = cv2.bitwise_not(thresh)
                images_to_try.append((170, thresh))
        else:
            # Dim LED - use OTSU auto-threshold
            if self._last_successful_thresh != 'otsu':
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                # Invert LED displays to get black text on white (standard OCR format)
                if is_light_on_dark:
                    thresh = cv2.bitwise_not(thresh)
                images_to_try.append(('otsu', thresh))

        # Fallback: add one additional method if primary fails
        if len(images_to_try) < 2:
            _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # Invert LED displays to get black text on white (standard OCR format)
            if is_light_on_dark:
                thresh = cv2.bitwise_not(thresh)
            images_to_try.append((150, thresh))

        # Try each preprocessed image with early exit
        all_raw_texts = []
        for thresh_id, img in images_to_try:
            tc, raw = self._ocr_timecode_easyocr(img)
            if tc is not None:
                # Cache successful threshold method for next frame
                self._last_successful_thresh = thresh_id
                # Update ROI cache with successful result
                result = (tc, raw)
                self._roi_cache = {'hash': roi_hash, 'result': result}
                return result
            if raw:
                all_raw_texts.append(raw)

        # Clear cache on failure (conditions may have changed)
        self._last_successful_thresh = None

        # Return best raw text even if parsing failed
        result = (None, all_raw_texts[0] if all_raw_texts else "")
        # Cache the result even on failure (frame may be genuinely unreadable)
        self._roi_cache = {'hash': roi_hash, 'result': result}
        return result
    
    def _ocr_timecode_easyocr(self, img: np.ndarray) -> Tuple[Optional[Timecode], str]:
        """
        Perform OCR using EasyOCR and parse timecode.

        OPTIMISED:
        - 4x scaling for better dot-matrix recognition
        - EasyOCR accepts grayscale directly (no BGR conversion needed)
        - Improved character correction mappings
        """
        try:
            reader = get_ocr_reader()

            # OPTIMIZATION: Scale up 4x for better dot-matrix digit recognition
            # Larger scale helps EasyOCR distinguish individual LED dots as solid digits
            scale = 4
            img_scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # OPTIMIZATION: EasyOCR accepts grayscale images directly
            # No need to convert to BGR - saves memory allocation and copy
            img_for_ocr = img_scaled

            img_height = img_for_ocr.shape[0]

            # Run EasyOCR - allowlist digits, colon, and common misread characters
            # Adding ; helps catch colon misreads
            results = reader.readtext(
                img_for_ocr,
                allowlist='0123456789:;.',
                paragraph=False,
                detail=1,
                width_ths=0.8,  # Slightly more lenient for grouped digits
                height_ths=0.8,
                low_text=0.3,   # Lower threshold to catch dim digits
            )

            if not results:
                return None, ""

            # Filter results: only keep text from the UPPER portion of the ROI
            # The timecode display is at the top, labels like SCENE/SHOT are below
            upper_threshold = img_height * 0.5  # Only look at top 50% of ROI

            filtered_results = []
            for (bbox, text, confidence) in results:
                # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                center_y = (bbox[0][1] + bbox[2][1]) / 2

                # Only include if in upper portion of image
                if center_y < upper_threshold and confidence > 0.1:
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

            # IMPROVED: Comprehensive character correction for dot-matrix displays
            # Normalize separators first
            raw_text = raw_text.replace(';', ':').replace('.', ':').replace(' ', '')

            # Common letter-to-digit misreads (expanded set)
            char_corrections = {
                'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'C': '0',
                'l': '1', 'I': '1', '|': '1', 'T': '1', 'i': '1', 'L': '1', '!': '1', 'J': '1',
                'Z': '2', 'z': '2',
                'E': '3',
                'A': '4', 'H': '4', 'h': '4',
                'S': '5', 's': '5',
                'G': '6', 'b': '6',
                'B': '8',
                'g': '9', 'q': '9', 'P': '9', 'p': '9',
            }
            for char, digit in char_corrections.items():
                raw_text = raw_text.replace(char, digit)

            # Try to parse timecode
            tc = Timecode.from_string(raw_text)
            return tc, raw_text

        except Exception as e:
            self.logger.debug(f"EasyOCR error: {e}")
            return None, ""

    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute perceptual hash for frame comparison."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self._hash_size, self._hash_size), interpolation=cv2.INTER_AREA)

        avg = np.mean(small)
        hash_bits = (small > avg).flatten()

        hash_int = 0
        for bit in hash_bits:
            hash_int = (hash_int << 1) | int(bit)

        return format(hash_int, f'0{self._hash_size * self._hash_size // 4}x')

    def check_timecode_duplicate(self, current_tc: Optional['Timecode']) -> bool:
        """
        Check if this frame is a duplicate based on TIMECODE VALUE.

        At 60fps with 30fps timecode, each timecode value appears for exactly 2 frames.
        Returns True if we've seen this timecode MORE times than expected (indicating a drop).
        """
        if current_tc is None:
            return False

        if self._prev_timecode is None:
            self._prev_timecode = current_tc
            self._consecutive_same_tc = 1
            return False

        if current_tc.to_frames() == self._prev_timecode.to_frames():
            self._consecutive_same_tc += 1
        else:
            self._prev_timecode = current_tc
            self._consecutive_same_tc = 1

        return self._consecutive_same_tc > self.frame_rate_multiplier


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

        self.stats = {
            'total_frames': 0,
            'successful_drops': 0,
            'false_negatives': 0,
            'false_positives': 0,
            'corruptions': 0,
            'undetected_skips': 0,
            'ocr_failures': 0,
            'physical_drops': 0,
            'indicated_drops': 0
        }

        self.logger = logging.getLogger(f"VideoAnalyser_{camera_id}")

        if DEBUG:
            DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)

    def load(self) -> bool:
        """Load video and metadata"""
        try:
            self.metadata = CameraMetadata.from_json_file(self.metadata_path)
            self.logger.info(f"Loaded metadata: {self.metadata.filename}")
            self.logger.info(f"  Reported drops: {self.metadata.dropped_frames}")
            self.logger.info(f"  Start TC: {self.metadata.start_timecode}")
            self.logger.info(f"  Frame rate: {self.metadata.frame_rate} fps")
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return False

        multiplier = compute_frame_rate_multiplier(self.metadata.frame_rate)
        self.logger.info(f"  Frame rate multiplier: {multiplier}")

        self.reader = FrameReader(self.video_path, self.camera_id, frame_rate_multiplier=multiplier)
        if not self.reader.open():
            return False

        return True

    def calibrate(self, manual_roi: Optional[Dict[str, int]] = None, save_debug: bool = False) -> bool:
        """Calibrate timecode detection on first frame."""
        if not self.reader:
            return False

        if manual_roi:
            self.reader.roi_timecode = manual_roi
            self.reader.timecode_calibrated = True
            self.logger.info(f"Using manual timecode ROI: {manual_roi}")
            return True

        config_roi = self._load_roi_from_config()
        if config_roi:
            self.reader.roi_timecode = config_roi
            self.reader.timecode_calibrated = True
            self.logger.info(f"Loaded timecode ROI from config: {config_roi}")
            return True

        frame = self.reader.read_frame(0)
        if frame is None:
            self.logger.error("Cannot read first frame for calibration")
            return False

        self.reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success = self.reader.auto_detect_timecode_roi(frame)
        if not success:
            self.logger.warning("Timecode ROI auto-detection failed, using fallback")
            self.logger.warning("Consider running: python calibrate_roi.py /path/to/video.mov")

        if save_debug and self.reader.roi_timecode:
            self._save_calibration_debug(frame)

        return True

    def _load_roi_from_config(self) -> Optional[Dict[str, int]]:
        """
        Load ROI from roi_config.json in the take folder.

        NOTE: ROI config is now per-take only (not inherited from shot folder)
        because the slate may move between takes.
        """
        video_folder = self.video_path.parent

        # Only look in the take folder - not parent folders
        # Each take requires its own ROI calibration
        config_path = video_folder / 'roi_config.json'

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # New format with per-camera ROIs
                camera_key = f'roi_{self.camera_id.lower()}'
                if camera_key in config:
                    roi = config[camera_key]
                    if roi and all(k in roi for k in ['x', 'y', 'width', 'height']):
                        self.logger.info(f"Found per-camera ROI config: {config_path} [{camera_key}]")
                        return roi

                # Fallback to roi_a/roi_b format
                if 'roi_a' in config and self.camera_id.upper() == 'A':
                    roi = config['roi_a']
                    if roi and all(k in roi for k in ['x', 'y', 'width', 'height']):
                        self.logger.info(f"Found ROI config: {config_path} [roi_a]")
                        return roi
                elif 'roi_b' in config and self.camera_id.upper() == 'B':
                    roi = config['roi_b']
                    if roi and all(k in roi for k in ['x', 'y', 'width', 'height']):
                        self.logger.info(f"Found ROI config: {config_path} [roi_b]")
                        return roi

                # Legacy format with single timecode_roi
                roi = config.get('timecode_roi')
                if roi and all(k in roi for k in ['x', 'y', 'width', 'height']):
                    self.logger.info(f"Found ROI config (legacy format): {config_path}")
                    return roi

            except Exception as e:
                self.logger.warning(f"Error loading ROI config {config_path}: {e}")

        return None

    def _save_calibration_debug(self, frame: np.ndarray):
        """Save a debug image showing the detected ROI regions"""
        debug_frame = frame.copy()

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

        DEBUG_OUTPUT_DIR.mkdir(exist_ok=True)
        debug_path = DEBUG_OUTPUT_DIR / f"calibration_cam{self.camera_id}.png"
        cv2.imwrite(str(debug_path), debug_frame)
        self.logger.info(f"Saved calibration debug image: {debug_path}")

    def analyse(self, progress_callback=None) -> bool:
        """Run full frame-by-frame analysis implementing Truth Table logic."""
        if not self.reader or not self.metadata:
            self.logger.error("Must call load() before analyse()")
            return False

        total = self.reader.total_frames
        self.stats['total_frames'] = total

        prev_analysis: Optional[FrameAnalysis] = None

        self.logger.info(f"Starting analysis of {total} frames...")

        start_time = datetime.now()
        last_progress_pct = -1

        self.reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_num in range(total):
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

            analysis = FrameAnalysis(frame_number=frame_num)

            # Check drop indicator
            analysis.drop_indicator_present = self.reader.detect_drop_indicator(frame)
            if analysis.drop_indicator_present:
                self.stats['indicated_drops'] += 1

            # Extract timecode via OCR
            tc, raw = self.reader.extract_timecode(frame)
            analysis.visual_timecode = tc
            analysis.visual_timecode_raw = raw

            # Timecode-based duplicate detection
            analysis.is_physical_duplicate = self.reader.check_timecode_duplicate(tc)
            if analysis.is_physical_duplicate:
                self.stats['physical_drops'] += 1

            # Compute frame hash
            analysis.image_hash = self.reader.compute_frame_hash(frame)

            # Apply Truth Table Logic
            analysis.status = self._apply_logic_gates(analysis, prev_analysis)

            # Update statistics
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
            prev_analysis = analysis

            # Progress reporting
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

            if progress_callback and frame_num % 100 == 0:
                progress_callback(frame_num, total)

        print()

        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Analysis complete in {elapsed:.1f}s ({total/elapsed:.1f} fps average)")

        # Print OCR statistics
        ocr_stats = self.reader.ocr_pipeline.stats
        success_rate = (ocr_stats['successes'] / ocr_stats['attempts'] * 100) if ocr_stats['attempts'] > 0 else 0
        self.logger.info(f"OCR Stats: {ocr_stats['successes']}/{ocr_stats['attempts']} ({success_rate:.1f}% success)")
        if ocr_stats['tesseract_wins'] > 0 or ocr_stats['easyocr_wins'] > 0:
            self.logger.info(f"  Tesseract wins: {ocr_stats['tesseract_wins']}, EasyOCR wins: {ocr_stats['easyocr_wins']}")

        return True

    def _apply_logic_gates(
        self,
        current: FrameAnalysis,
        previous: Optional[FrameAnalysis]
    ) -> FrameStatus:
        """Apply the Truth Table logic gates to determine frame status."""
        # Logic Gate 2: Drop Frame Verification
        if current.is_physical_duplicate and current.drop_indicator_present:
            return FrameStatus.DROPPED_DETECTED

        if current.is_physical_duplicate and not current.drop_indicator_present:
            current.notes = "Physical duplicate without indicator"
            return FrameStatus.DROPPED_UNDETECTED

        if not current.is_physical_duplicate and current.drop_indicator_present:
            current.notes = "Indicator present but no physical duplicate"
            return FrameStatus.FALSE_POSITIVE

        # Logic Gate 1: Continuity & Corruption
        if previous and previous.visual_timecode and current.visual_timecode:
            prev_tc_frames = previous.visual_timecode.to_frames(
                self.metadata.frame_rate if self.metadata else 60
            )
            curr_tc_frames = current.visual_timecode.to_frames(
                self.metadata.frame_rate if self.metadata else 60
            )

            actual_diff = curr_tc_frames - prev_tc_frames

            if abs(actual_diff) > TIMECODE_JUMP_THRESHOLD:
                current.notes = f"Timecode jump: {previous.visual_timecode} -> {current.visual_timecode}"
                return FrameStatus.CORRUPTION

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
        """Get the timecode from the start of the video."""
        if not self.results:
            return None

        search_limit = min(50, len(self.results))
        for i in range(search_limit):
            if self.results[i].visual_timecode is not None:
                tc = self.results[i].visual_timecode
                multiplier = self.reader.frame_rate_multiplier if self.reader else 2
                tc_frames_offset = i // multiplier

                total_frames = tc.to_frames() - tc_frames_offset
                if total_frames >= 0:
                    fps = self.metadata.frame_rate if self.metadata else 60
                    return Timecode.from_frame_number(total_frames, fps)
                else:
                    return tc

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

        self._roi_a = roi_a
        self._roi_b = roi_b

        self.validation_results: Dict[str, Any] = {}
        self.logger = logging.getLogger("DualCameraValidator")

    def run_full_validation(self, progress_callback=None, save_calibration_debug: bool = True) -> bool:
        """Execute complete validation pipeline"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Dual Camera Validation")
        self.logger.info("=" * 60)

        if not self.analyser_a.load():
            self.logger.error("Failed to load Camera A")
            return False

        if not self.analyser_b.load():
            self.logger.error("Failed to load Camera B")
            return False

        self.analyser_a.calibrate(save_debug=save_calibration_debug, manual_roi=self._roi_a)
        self.analyser_b.calibrate(save_debug=save_calibration_debug, manual_roi=self._roi_b)

        print("\n  Analysing Camera A (Left)...")
        self.analyser_a.analyse(progress_callback)

        print("\n  Analysing Camera B (Right)...")
        self.analyser_b.analyse(progress_callback)

        self._validate_sync()
        self._validate_corruption()
        self._validate_drop_accuracy()
        self._validate_start_timecode()

        self.analyser_a.cleanup()
        self.analyser_b.cleanup()

        return True

    def _validate_sync(self):
        """Validate dual-camera synchronisation."""
        tc_a = self.analyser_a.get_frame_zero_timecode()
        tc_b = self.analyser_b.get_frame_zero_timecode()

        meta_tc_a = self.analyser_a.metadata.start_timecode if self.analyser_a.metadata else None
        meta_tc_b = self.analyser_b.metadata.start_timecode if self.analyser_b.metadata else None

        if tc_a and tc_b:
            sync_pass = tc_a == tc_b
            source = "OCR"
        elif meta_tc_a and meta_tc_b:
            sync_pass = meta_tc_a == meta_tc_b
            source = "metadata"
            tc_a = meta_tc_a
            tc_b = meta_tc_b
        else:
            sync_pass = False
            source = "unknown"

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

            drop_match = metadata_drops == detected_drops

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
        """Validate that OCR-detected start timecode matches metadata."""
        results = {}

        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if not analyser.metadata:
                continue

            ocr_tc = analyser.get_frame_zero_timecode()
            metadata_tc = analyser.metadata.start_timecode

            if ocr_tc and metadata_tc:
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
                match = True
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
        """Generate validation report in both text and HTML formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Text report
        text_lines = self._generate_text_report()
        text_path = self.output_dir / f"qa_report_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write('\n'.join(text_lines))

        # HTML report
        html_content = self._generate_html_report()
        html_path = self.output_dir / f"qa_report_{timestamp}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

        # JSON report
        json_path = self.output_dir / f"qa_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'validation_results': self.validation_results,
                'camera_a_stats': self.analyser_a.stats,
                'camera_b_stats': self.analyser_b.stats,
                'camera_a_metadata': {
                    'filename': self.analyser_a.metadata.filename if self.analyser_a.metadata else '',
                    'frame_rate': self.analyser_a.metadata.frame_rate if self.analyser_a.metadata else 0,
                    'num_frames': self.analyser_a.metadata.num_frames if self.analyser_a.metadata else 0,
                    'dropped_frames': self.analyser_a.metadata.dropped_frames if self.analyser_a.metadata else 0,
                    'start_timecode': str(self.analyser_a.metadata.start_timecode) if self.analyser_a.metadata else ''
                },
                'camera_b_metadata': {
                    'filename': self.analyser_b.metadata.filename if self.analyser_b.metadata else '',
                    'frame_rate': self.analyser_b.metadata.frame_rate if self.analyser_b.metadata else 0,
                    'num_frames': self.analyser_b.metadata.num_frames if self.analyser_b.metadata else 0,
                    'dropped_frames': self.analyser_b.metadata.dropped_frames if self.analyser_b.metadata else 0,
                    'start_timecode': str(self.analyser_b.metadata.start_timecode) if self.analyser_b.metadata else ''
                }
            }, f, indent=2)

        self.logger.info(f"Reports saved to: {self.output_dir}")

        return str(text_path), str(html_path)

    def _generate_text_report(self) -> List[str]:
        """Generate text format report"""
        lines = [
            "=" * 80,
            "VIDEO QA VALIDATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 80,
            "CAMERA INFORMATION",
            "-" * 80
        ]

        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if analyser.metadata:
                lines.extend([
                    f"Camera {cam_id}: {analyser.metadata.filename}",
                    f"  Frames: {analyser.metadata.num_frames}",
                    f"  Frame Rate: {analyser.metadata.frame_rate} fps",
                    f"  Start TC: {analyser.metadata.start_timecode}",
                    ""
                ])

        # Sync status
        lines.extend([
            "-" * 80,
            "1. SYNC STATUS",
            "-" * 80
        ])
        sync = self.validation_results.get('sync', {})
        lines.extend([
            f"Status: {sync.get('status', 'N/A')}",
            f"Camera A Frame 0 TC: {sync.get('camera_a_frame0_tc', 'N/A')}",
            f"Camera B Frame 0 TC: {sync.get('camera_b_frame0_tc', 'N/A')}",
            f"Notes: {sync.get('notes', '')}",
            ""
        ])

        # Corruption status
        lines.extend([
            "-" * 80,
            "2. CORRUPTION STATUS",
            "-" * 80
        ])
        corruption = self.validation_results.get('corruption', {})
        for cam_id in ['A', 'B']:
            cam_data = corruption.get(cam_id, {})
            lines.extend([
                f"Camera {cam_id}:",
                f"  Status: {cam_data.get('status', 'N/A')}",
                f"  Metadata Corrupted At: {cam_data.get('metadata_corrupted_at', 'N/A')}",
                f"  Vision Corruptions: {cam_data.get('vision_corruptions_detected', 'N/A')}",
                f"  Notes: {cam_data.get('notes', '')}",
                ""
            ])

        # Drop accuracy
        lines.extend([
            "-" * 80,
            "3. DROP FRAME ACCURACY",
            "-" * 80,
            f"{'Metric':<35} {'Camera A':>15} {'Camera B':>15}",
            "-" * 65
        ])
        drop_data = self.validation_results.get('drop_accuracy', {})
        metrics = [
            ('metadata_drop_count', 'Metadata Drop Count'),
            ('physical_drops_detected', 'Physical Drops Detected'),
            ('indicator_drops_counted', 'Indicator Drops Counted'),
            ('successful_processes', 'Successful Processes'),
            ('false_negatives', 'False Negatives'),
            ('false_positives', 'False Positives'),
            ('drop_count_match', 'Drop Count Match'),
            ('indicator_accuracy', 'Indicator Accuracy')
        ]
        for key, label in metrics:
            val_a = drop_data.get('A', {}).get(key, 'N/A')
            val_b = drop_data.get('B', {}).get(key, 'N/A')
            lines.append(f"{label:<35} {str(val_a):>15} {str(val_b):>15}")
        lines.append("")

        # Start timecode
        lines.extend([
            "-" * 80,
            "4. START TIMECODE VALIDATION",
            "-" * 80
        ])
        start_tc = self.validation_results.get('start_timecode', {})
        for cam_id in ['A', 'B']:
            cam_data = start_tc.get(cam_id, {})
            lines.extend([
                f"Camera {cam_id}:",
                f"  Status: {cam_data.get('status', 'N/A')}",
                f"  OCR Frame 0: {cam_data.get('ocr_frame0_tc', 'N/A')}",
                f"  Metadata: {cam_data.get('metadata_start_tc', 'N/A')}",
                f"  Notes: {cam_data.get('notes', '')}",
                ""
            ])

        # Overall result
        overall_pass = all([
            sync.get('status') == 'PASS',
            all(c.get('status') == 'PASS' for c in corruption.values()),
            all(d.get('drop_count_match') == 'PASS' for d in drop_data.values()),
            all(s.get('status') == 'PASS' for s in start_tc.values())
        ])

        lines.extend([
            "=" * 80,
            f"OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}",
            "=" * 80
        ])

        return lines

    def _generate_html_report(self) -> str:
        """Generate HTML format report"""
        sync = self.validation_results.get('sync', {})
        corruption = self.validation_results.get('corruption', {})
        drop_data = self.validation_results.get('drop_accuracy', {})
        start_tc = self.validation_results.get('start_timecode', {})

        overall_pass = all([
            sync.get('status') == 'PASS',
            all(c.get('status') == 'PASS' for c in corruption.values()),
            all(d.get('drop_count_match') == 'PASS' for d in drop_data.values()),
            all(s.get('status') == 'PASS' for s in start_tc.values())
        ])

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video QA Report</title>
    <style>
        :root {{
            --pass-color: #28a745;
            --fail-color: #dc3545;
            --warning-color: #ffc107;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        h1, h2 {{ color: #343a40; }}
        .status-pass {{ color: var(--pass-color); font-weight: bold; }}
        .status-fail {{ color: var(--fail-color); font-weight: bold; }}
        .status-warning {{ color: var(--warning-color); font-weight: bold; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{ background-color: #343a40; color: white; }}
        tr:hover {{ background-color: #f1f3f4; }}
        .overall-result {{
            text-align: center;
            padding: 30px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 24px;
            font-weight: bold;
        }}
        .overall-pass {{ background-color: #d4edda; color: #155724; }}
        .overall-fail {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>Video QA Validation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="overall-result {'overall-pass' if overall_pass else 'overall-fail'}">
        OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}
    </div>

    <div class="card">
        <h2>1. Sync Status</h2>
        <p>Status: <span class="status-{sync.get('status', 'fail').lower()}">{sync.get('status', 'N/A')}</span></p>
        <table>
            <tr><td>Camera A Frame 0 TC</td><td>{sync.get('camera_a_frame0_tc', 'N/A')}</td></tr>
            <tr><td>Camera B Frame 0 TC</td><td>{sync.get('camera_b_frame0_tc', 'N/A')}</td></tr>
            <tr><td>Notes</td><td>{sync.get('notes', '')}</td></tr>
        </table>
    </div>

    <div class="card">
        <h2>2. Corruption Status</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Camera A</th>
                <th>Camera B</th>
            </tr>
            <tr>
                <td>Status</td>
                <td class="status-{corruption.get('A', {}).get('status', 'fail').lower()}">{corruption.get('A', {}).get('status', 'N/A')}</td>
                <td class="status-{corruption.get('B', {}).get('status', 'fail').lower()}">{corruption.get('B', {}).get('status', 'N/A')}</td>
            </tr>
            <tr>
                <td>Metadata Corrupted At</td>
                <td>{corruption.get('A', {}).get('metadata_corrupted_at', 'N/A')}</td>
                <td>{corruption.get('B', {}).get('metadata_corrupted_at', 'N/A')}</td>
            </tr>
            <tr>
                <td>Vision Corruptions</td>
                <td>{corruption.get('A', {}).get('vision_corruptions_detected', 'N/A')}</td>
                <td>{corruption.get('B', {}).get('vision_corruptions_detected', 'N/A')}</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>3. Drop Frame Accuracy</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Camera A</th>
                <th>Camera B</th>
            </tr>
            <tr>
                <td>Metadata Drop Count</td>
                <td>{drop_data.get('A', {}).get('metadata_drop_count', 'N/A')}</td>
                <td>{drop_data.get('B', {}).get('metadata_drop_count', 'N/A')}</td>
            </tr>
            <tr>
                <td>Physical Drops Detected</td>
                <td>{drop_data.get('A', {}).get('physical_drops_detected', 'N/A')}</td>
                <td>{drop_data.get('B', {}).get('physical_drops_detected', 'N/A')}</td>
            </tr>
            <tr>
                <td>Indicator Drops Counted</td>
                <td>{drop_data.get('A', {}).get('indicator_drops_counted', 'N/A')}</td>
                <td>{drop_data.get('B', {}).get('indicator_drops_counted', 'N/A')}</td>
            </tr>
            <tr>
                <td>Successful Processes</td>
                <td>{drop_data.get('A', {}).get('successful_processes', 'N/A')}</td>
                <td>{drop_data.get('B', {}).get('successful_processes', 'N/A')}</td>
            </tr>
            <tr>
                <td>False Negatives</td>
                <td>{drop_data.get('A', {}).get('false_negatives', 'N/A')}</td>
                <td>{drop_data.get('B', {}).get('false_negatives', 'N/A')}</td>
            </tr>
            <tr>
                <td>False Positives</td>
                <td>{drop_data.get('A', {}).get('false_positives', 'N/A')}</td>
                <td>{drop_data.get('B', {}).get('false_positives', 'N/A')}</td>
            </tr>
            <tr>
                <td>Drop Count Match</td>
                <td class="status-{drop_data.get('A', {}).get('drop_count_match', 'fail').lower()}">{drop_data.get('A', {}).get('drop_count_match', 'N/A')}</td>
                <td class="status-{drop_data.get('B', {}).get('drop_count_match', 'fail').lower()}">{drop_data.get('B', {}).get('drop_count_match', 'N/A')}</td>
            </tr>
            <tr>
                <td>Indicator Accuracy</td>
                <td class="status-{drop_data.get('A', {}).get('indicator_accuracy', 'fail').lower()}">{drop_data.get('A', {}).get('indicator_accuracy', 'N/A')}</td>
                <td class="status-{drop_data.get('B', {}).get('indicator_accuracy', 'fail').lower()}">{drop_data.get('B', {}).get('indicator_accuracy', 'N/A')}</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>4. Start Timecode Validation</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Camera A</th>
                <th>Camera B</th>
            </tr>
            <tr>
                <td>Status</td>
                <td class="status-{start_tc.get('A', {}).get('status', 'fail').lower()}">{start_tc.get('A', {}).get('status', 'N/A')}</td>
                <td class="status-{start_tc.get('B', {}).get('status', 'fail').lower()}">{start_tc.get('B', {}).get('status', 'N/A')}</td>
            </tr>
            <tr>
                <td>OCR Frame 0</td>
                <td>{start_tc.get('A', {}).get('ocr_frame0_tc', 'N/A')}</td>
                <td>{start_tc.get('B', {}).get('ocr_frame0_tc', 'N/A')}</td>
            </tr>
            <tr>
                <td>Metadata</td>
                <td>{start_tc.get('A', {}).get('metadata_start_tc', 'N/A')}</td>
                <td>{start_tc.get('B', {}).get('metadata_start_tc', 'N/A')}</td>
            </tr>
        </table>
    </div>
</body>
</html>'''

        return html


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging for the application"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Video QA Validation - Process dual-camera recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--video-a', '-va', required=True, help='Path to Camera A video file')
    parser.add_argument('--video-b', '-vb', required=True, help='Path to Camera B video file')
    parser.add_argument('--metadata-a', '-ma', required=True, help='Path to Camera A metadata JSON')
    parser.add_argument('--metadata-b', '-mb', required=True, help='Path to Camera B metadata JSON')
    parser.add_argument('--output', '-o', default='qa_reports', help='Output directory for reports')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    setup_logging(args.verbose)

    global DEBUG
    DEBUG = args.debug

    validator = DualCameraValidator(
        video_a=Path(args.video_a),
        video_b=Path(args.video_b),
        metadata_a=Path(args.metadata_a),
        metadata_b=Path(args.metadata_b),
        output_dir=Path(args.output)
    )

    success = validator.run_full_validation()

    if success:
        text_path, html_path = validator.generate_report()
        print(f"\nReports generated:")
        print(f"  Text: {text_path}")
        print(f"  HTML: {html_path}")
        return 0
    else:
        print("\nValidation failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
