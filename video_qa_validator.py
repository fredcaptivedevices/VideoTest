#!/usr/bin/env python3
"""
Video QA Automation Application for Hardware Validation
========================================================
Processes dual-stream video (Camera A/B) and metadata to validate recording integrity.

Author: Captive Devices QA System
Version: 1.0.0
"""

import cv2
import numpy as np
import pandas as pd
import json
import re
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from datetime import datetime
from enum import Enum, auto
import pytesseract
from collections import defaultdict
import imagehash
from PIL import Image


# =============================================================================
# Configuration & Constants
# =============================================================================

DEBUG = True  # Set to True to save screenshots of flagged frames
DEBUG_OUTPUT_DIR = Path("debug_frames")

# Drop indicator ROI - Fixed position based on screenshot analysis
# The "Dropped frame" text appears in top-left corner
ROI_DROP = {
    'x': 0,
    'y': 0,
    'width': 250,
    'height': 50
}

# Timecode pattern for OCR validation
TIMECODE_PATTERN = re.compile(r'(\d{1,2}):(\d{2}):(\d{2}):(\d{2})')
TIMECODE_LOOSE_PATTERN = re.compile(r'(\d{1,2})\D*(\d{2})\D*(\d{2})\D*(\d{2})')

# Image comparison thresholds
DUPLICATE_FRAME_THRESHOLD = 5  # Hash difference threshold for duplicate detection
TIMECODE_JUMP_THRESHOLD = 100  # Maximum expected frame jump before flagging corruption


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
        """Parse timecode from string format HH:MM:SS:FF"""
        match = TIMECODE_PATTERN.search(tc_string)
        if not match:
            match = TIMECODE_LOOSE_PATTERN.search(tc_string)
        if match:
            return cls(
                hours=int(match.group(1)),
                minutes=int(match.group(2)),
                seconds=int(match.group(3)),
                frames=int(match.group(4))
            )
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
    """
    
    def __init__(self, video_path: Path, camera_id: str = "A"):
        self.video_path = Path(video_path)
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        
        # ROI for timecode - auto-detected on first frame
        self.roi_timecode: Optional[Dict[str, int]] = None
        self.timecode_calibrated: bool = False
        
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
        
        self.logger.info(f"Opened video: {self.video_path.name}")
        self.logger.info(f"  Frames: {self.total_frames}, FPS: {self.fps}")
        self.logger.info(f"  Resolution: {self.frame_width}x{self.frame_height}")
        
        return True
    
    def close(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
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
        
        # Apply threshold to isolate white text on dark background
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
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
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold for better digit detection
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Also try inverse for LED displays (bright digits on dark)
        _, thresh_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Scan regions of the image for timecode pattern
        scan_regions = self._generate_scan_regions(frame.shape)
        
        for region in scan_regions:
            x, y, w, h = region
            
            # Try both threshold methods
            for thresh_img in [thresh, thresh_inv]:
                roi = thresh_img[y:y+h, x:x+w]
                
                # OCR with digit-focused config
                ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789:'
                text = pytesseract.image_to_string(roi, config=ocr_config).strip()
                
                # Check if we found a timecode pattern
                if TIMECODE_PATTERN.search(text) or TIMECODE_LOOSE_PATTERN.search(text):
                    # Expand ROI slightly for safety margin
                    self.roi_timecode = {
                        'x': max(0, x - 20),
                        'y': max(0, y - 20),
                        'width': min(w + 40, frame.shape[1] - x),
                        'height': min(h + 40, frame.shape[0] - y)
                    }
                    self.timecode_calibrated = True
                    self.logger.info(f"Timecode ROI detected: {self.roi_timecode}")
                    return True
        
        # If auto-detection fails, try full-frame OCR as fallback
        self.logger.warning("Auto-detection failed, using full-frame scan")
        self.roi_timecode = {
            'x': 0,
            'y': 0,
            'width': frame.shape[1],
            'height': frame.shape[0]
        }
        return False
    
    def _generate_scan_regions(self, shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        """Generate regions to scan for timecode display"""
        height, width = shape[:2]
        regions = []
        
        # Common positions for digital slates
        # Centre-right area (where clapperboard typically is)
        regions.append((width // 3, height // 4, width * 2 // 3, height // 2))
        
        # Full centre
        regions.append((width // 4, height // 4, width // 2, height // 2))
        
        # Right third
        regions.append((width * 2 // 3, 0, width // 3, height))
        
        # Sliding window approach for thorough coverage
        window_w, window_h = 400, 100
        step = 100
        
        for y in range(0, height - window_h, step):
            for x in range(0, width - window_w, step):
                regions.append((x, y, window_w, window_h))
        
        return regions
    
    def extract_timecode(self, frame: np.ndarray) -> Tuple[Optional[Timecode], str]:
        """
        Extract timecode from frame using OCR.
        Returns tuple of (Timecode object, raw OCR text).
        """
        if not self.roi_timecode:
            return None, ""
        
        # Extract ROI
        roi = frame[
            self.roi_timecode['y']:self.roi_timecode['y'] + self.roi_timecode['height'],
            self.roi_timecode['x']:self.roi_timecode['x'] + self.roi_timecode['width']
        ]
        
        # Pre-process for LED display (white/green digits on black)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Multiple threshold attempts
        results = []
        
        # Method 1: Simple binary threshold (for bright displays)
        _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        results.append(self._ocr_timecode(thresh1))
        
        # Method 2: OTSU threshold
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(self._ocr_timecode(thresh2))
        
        # Method 3: Adaptive threshold
        thresh3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results.append(self._ocr_timecode(thresh3))
        
        # Method 4: Inverted (for dark text on light background)
        _, thresh4 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        results.append(self._ocr_timecode(thresh4))
        
        # Return the first successful parse
        for tc, raw in results:
            if tc is not None:
                return tc, raw
        
        # Return best raw text even if parsing failed
        raw_texts = [r[1] for r in results if r[1]]
        return None, raw_texts[0] if raw_texts else ""
    
    def _ocr_timecode(self, thresh_img: np.ndarray) -> Tuple[Optional[Timecode], str]:
        """Perform OCR on thresholded image and parse timecode"""
        # Scale up for better OCR
        scaled = cv2.resize(thresh_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # OCR configuration for seven-segment displays
        ocr_config = '--psm 7 -c tessedit_char_whitelist=0123456789:.'
        raw_text = pytesseract.image_to_string(scaled, config=ocr_config).strip()
        
        # Clean up common OCR errors
        raw_text = raw_text.replace('.', ':').replace(' ', '')
        
        tc = Timecode.from_string(raw_text)
        return tc, raw_text
    
    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute perceptual hash for frame comparison"""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Compute average hash (fast and effective for duplicate detection)
        hash_value = imagehash.average_hash(pil_image)
        return str(hash_value)
    
    def frames_are_duplicate(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        Check if two frames are visual duplicates (physical drop).
        Uses perceptual hashing to handle minor compression artefacts.
        """
        hash1 = imagehash.average_hash(Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)))
        hash2 = imagehash.average_hash(Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)))
        
        # Hamming distance between hashes
        difference = hash1 - hash2
        return difference <= DUPLICATE_FRAME_THRESHOLD


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
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return False
        
        # Open video
        self.reader = FrameReader(self.video_path, self.camera_id)
        if not self.reader.open():
            return False
        
        return True
    
    def calibrate(self) -> bool:
        """Calibrate timecode detection on first frame"""
        if not self.reader:
            return False
        
        frame = self.reader.read_frame(0)
        if frame is None:
            self.logger.error("Cannot read first frame for calibration")
            return False
        
        success = self.reader.auto_detect_timecode_roi(frame)
        if not success:
            self.logger.warning("Timecode ROI auto-detection failed, using fallback")
        
        return True
    
    def analyse(self, progress_callback=None) -> bool:
        """
        Run full frame-by-frame analysis implementing Truth Table logic.
        """
        if not self.reader or not self.metadata:
            self.logger.error("Must call load() before analyse()")
            return False
        
        total = self.reader.total_frames
        self.stats['total_frames'] = total
        
        prev_frame: Optional[np.ndarray] = None
        prev_analysis: Optional[FrameAnalysis] = None
        
        self.logger.info(f"Starting analysis of {total} frames...")
        
        for frame_num in range(total):
            # Read current frame
            frame = self.reader.read_frame(frame_num)
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
            
            # Compute frame hash
            analysis.image_hash = self.reader.compute_frame_hash(frame)
            
            # Check drop indicator
            analysis.drop_indicator_present = self.reader.detect_drop_indicator(frame)
            if analysis.drop_indicator_present:
                self.stats['indicated_drops'] += 1
            
            # Extract timecode
            tc, raw = self.reader.extract_timecode(frame)
            analysis.visual_timecode = tc
            analysis.visual_timecode_raw = raw
            
            # Check for physical duplicate (dropped frame)
            if prev_frame is not None:
                analysis.is_physical_duplicate = self.reader.frames_are_duplicate(
                    prev_frame, frame
                )
                if analysis.is_physical_duplicate:
                    self.stats['physical_drops'] += 1
            
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
            prev_frame = frame.copy()
            prev_analysis = analysis
            
            # Progress callback
            if progress_callback and frame_num % 100 == 0:
                progress_callback(frame_num, total)
        
        self.logger.info("Analysis complete")
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
        """Get the timecode from frame 0"""
        if self.results and len(self.results) > 0:
            return self.results[0].visual_timecode
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
        output_dir: Path = Path("qa_reports")
    ):
        self.analyser_a = VideoAnalyser(video_a, metadata_a, "A")
        self.analyser_b = VideoAnalyser(video_b, metadata_b, "B")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.validation_results: Dict[str, Any] = {}
        self.logger = logging.getLogger("DualCameraValidator")
    
    def run_full_validation(self, progress_callback=None) -> bool:
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
        
        # Calibrate timecode detection
        self.analyser_a.calibrate()
        self.analyser_b.calibrate()
        
        # Run analysis on both cameras
        self.logger.info("Analysing Camera A...")
        self.analyser_a.analyse(progress_callback)
        
        self.logger.info("Analysing Camera B...")
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
        """Validate dual-camera synchronisation"""
        tc_a = self.analyser_a.get_frame_zero_timecode()
        tc_b = self.analyser_b.get_frame_zero_timecode()
        
        sync_pass = tc_a == tc_b if (tc_a and tc_b) else False
        
        self.validation_results['sync'] = {
            'status': 'PASS' if sync_pass else 'FAIL',
            'camera_a_frame0_tc': str(tc_a) if tc_a else 'N/A',
            'camera_b_frame0_tc': str(tc_b) if tc_b else 'N/A',
            'notes': '' if sync_pass else 'Frame 0 timecodes do not match'
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
        """Validate OCR frame 0 matches metadata start timecode"""
        results = {}
        
        for cam_id, analyser in [('A', self.analyser_a), ('B', self.analyser_b)]:
            if not analyser.metadata:
                continue
            
            ocr_tc = analyser.get_frame_zero_timecode()
            metadata_tc = analyser.metadata.start_timecode
            
            match = ocr_tc == metadata_tc if (ocr_tc and metadata_tc) else False
            
            results[cam_id] = {
                'status': 'PASS' if match else 'FAIL',
                'ocr_frame0_tc': str(ocr_tc) if ocr_tc else 'N/A',
                'metadata_start_tc': str(metadata_tc),
                'notes': '' if match else 'Start timecode mismatch'
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
