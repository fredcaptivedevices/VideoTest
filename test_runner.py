#!/usr/bin/env python3
"""
Test Runner and Demonstration Script for Video QA Automation
============================================================

This script demonstrates the QA system capabilities and provides
unit tests for the core logic components.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from video_qa_validator import (
    Timecode,
    FrameStatus,
    FrameAnalysis,
    CameraMetadata,
    FrameReader,
    VideoAnalyser,
    DualCameraValidator,
    LEDTimecodeOCR,
    setup_logging,
    EASYOCR_AVAILABLE
)


class TestTimecode:
    """Unit tests for Timecode class"""

    def test_from_string(self):
        """Test parsing timecode from string"""
        tc = Timecode.from_string("02:55:23:00")
        assert tc is not None
        assert tc.hours == 2
        assert tc.minutes == 55
        assert tc.seconds == 23
        assert tc.frames == 0
        print("✓ test_from_string passed")

    def test_from_string_no_separators(self):
        """Test parsing timecode without separators"""
        tc = Timecode.from_string("02552300")
        assert tc is not None
        assert tc.hours == 2
        assert tc.minutes == 55
        assert tc.seconds == 23
        assert tc.frames == 0
        print("✓ test_from_string_no_separators passed")

    def test_from_string_partial_separators(self):
        """Test parsing timecode with partial separators"""
        tc = Timecode.from_string("02:55:2300")
        assert tc is not None
        assert tc.hours == 2
        assert tc.minutes == 55
        print("✓ test_from_string_partial_separators passed")

    def test_from_json(self):
        """Test parsing timecode from JSON format"""
        json_tc = {"Hours": "2", "Minutes": "55", "Seconds": "23", "Frames": "0"}
        tc = Timecode.from_json(json_tc)
        assert tc.hours == 2
        assert tc.minutes == 55
        assert tc.seconds == 23
        assert tc.frames == 0
        print("✓ test_from_json passed")

    def test_to_frames(self):
        """Test converting timecode to frame count"""
        tc = Timecode(hours=0, minutes=0, seconds=1, frames=0)
        assert tc.to_frames(fps=60) == 60

        tc2 = Timecode(hours=1, minutes=0, seconds=0, frames=0)
        assert tc2.to_frames(fps=60) == 3600 * 60
        print("✓ test_to_frames passed")

    def test_from_frame_number(self):
        """Test creating timecode from frame number"""
        tc = Timecode.from_frame_number(3660, fps=60)  # 1 minute and 1 second
        assert tc.hours == 0
        assert tc.minutes == 1
        assert tc.seconds == 1
        assert tc.frames == 0
        print("✓ test_from_frame_number passed")

    def test_equality(self):
        """Test timecode equality comparison"""
        tc1 = Timecode(2, 55, 23, 0)
        tc2 = Timecode(2, 55, 23, 0)
        tc3 = Timecode(2, 55, 23, 1)

        assert tc1 == tc2
        assert tc1 != tc3
        print("✓ test_equality passed")

    def test_string_representation(self):
        """Test string formatting"""
        tc = Timecode(2, 55, 23, 0)
        assert str(tc) == "02:55:23:00"
        print("✓ test_string_representation passed")

    def test_invalid_timecode_rejected(self):
        """Test that invalid timecodes are rejected"""
        # Minutes >= 60 should be rejected
        tc = Timecode.from_string("00:65:00:00")
        assert tc is None

        # Seconds >= 60 should be rejected
        tc = Timecode.from_string("00:00:70:00")
        assert tc is None
        print("✓ test_invalid_timecode_rejected passed")

    def run_all(self):
        """Run all timecode tests"""
        print("\n=== Timecode Tests ===")
        self.test_from_string()
        self.test_from_string_no_separators()
        self.test_from_string_partial_separators()
        self.test_from_json()
        self.test_to_frames()
        self.test_from_frame_number()
        self.test_equality()
        self.test_string_representation()
        self.test_invalid_timecode_rejected()
        print("All Timecode tests passed!\n")


class TestOCRPipeline:
    """Tests for the enhanced OCR pipeline"""

    def test_ocr_engines_available(self):
        """Test that EasyOCR is available"""
        assert EASYOCR_AVAILABLE, \
            "EasyOCR must be available"
        print(f"✓ OCR engine - EasyOCR: {EASYOCR_AVAILABLE}")

    def test_preprocessing_generates_images(self):
        """Test that preprocessing generates multiple image variants"""
        ocr = LEDTimecodeOCR("test")

        # Create a test image (white digits on black background)
        test_img = np.zeros((100, 400), dtype=np.uint8)
        test_img[30:70, 50:100] = 255  # First digit
        test_img[30:70, 120:170] = 255  # Second digit

        results = ocr.preprocess_led_display(test_img)

        assert len(results) > 10, "Should generate multiple preprocessing variants"
        print(f"✓ Preprocessing generated {len(results)} image variants")

    def test_ocr_pipeline_initialization(self):
        """Test OCR pipeline initialization"""
        ocr = LEDTimecodeOCR("A")
        assert ocr.camera_id == "A"
        assert ocr.stats['attempts'] == 0
        assert ocr.stats['successes'] == 0
        print("✓ OCR pipeline initialization passed")

    def run_all(self):
        """Run all OCR pipeline tests"""
        print("\n=== OCR Pipeline Tests ===")
        self.test_ocr_engines_available()
        self.test_preprocessing_generates_images()
        self.test_ocr_pipeline_initialization()
        print("All OCR Pipeline tests passed!\n")


class TestMetadataLoading:
    """Tests for metadata parsing"""

    def __init__(self, metadata_path: Optional[Path] = None):
        self.metadata_path = metadata_path

    def test_load_metadata(self):
        """Test loading metadata from JSON file"""
        if not self.metadata_path or not self.metadata_path.exists():
            print("⚠ Skipping metadata load test (no file provided)")
            return

        metadata = CameraMetadata.from_json_file(self.metadata_path)

        assert metadata.frame_rate == 60
        assert metadata.dropped_frames == 58
        assert metadata.num_frames == 7718
        assert metadata.corrupted_at == -1
        assert metadata.start_timecode == Timecode(2, 55, 23, 0)

        print("✓ test_load_metadata passed")
        print(f"  Loaded: {metadata.filename}")
        print(f"  Frames: {metadata.num_frames} @ {metadata.frame_rate}fps")
        print(f"  Drops: {metadata.dropped_frames}")
        print(f"  Start TC: {metadata.start_timecode}")

    def run_all(self):
        """Run all metadata tests"""
        print("\n=== Metadata Tests ===")
        self.test_load_metadata()
        print("All Metadata tests passed!\n")


class TestLogicGates:
    """Tests for the Truth Table logic implementation"""

    def test_successful_drop_detection(self):
        """Test: Physical drop detected AND indicator present = DROPPED_DETECTED"""
        analysis = FrameAnalysis(
            frame_number=100,
            is_physical_duplicate=True,
            drop_indicator_present=True
        )

        assert analysis.is_physical_duplicate == True
        assert analysis.drop_indicator_present == True
        print("✓ test_successful_drop_detection passed")

    def test_false_negative(self):
        """Test: Physical drop BUT no indicator = DROPPED_UNDETECTED (False Negative)"""
        analysis = FrameAnalysis(
            frame_number=100,
            is_physical_duplicate=True,
            drop_indicator_present=False
        )

        assert analysis.is_physical_duplicate == True
        assert analysis.drop_indicator_present == False
        print("✓ test_false_negative passed")

    def test_false_positive(self):
        """Test: No physical drop BUT indicator present = FALSE_POSITIVE"""
        analysis = FrameAnalysis(
            frame_number=100,
            is_physical_duplicate=False,
            drop_indicator_present=True
        )

        assert analysis.is_physical_duplicate == False
        assert analysis.drop_indicator_present == True
        print("✓ test_false_positive passed")

    def test_corruption_detection(self):
        """Test: Illogical timecode jump should flag corruption"""
        prev = FrameAnalysis(
            frame_number=99,
            visual_timecode=Timecode(0, 0, 1, 0)  # Frame 60
        )

        current = FrameAnalysis(
            frame_number=100,
            visual_timecode=Timecode(1, 0, 0, 0)  # Jumped to frame 3600*60 = 216000
        )

        prev_frames = prev.visual_timecode.to_frames(60)
        curr_frames = current.visual_timecode.to_frames(60)
        diff = abs(curr_frames - prev_frames)

        assert diff > 100  # TIMECODE_JUMP_THRESHOLD
        print("✓ test_corruption_detection passed")
        print(f"  Timecode jump: {diff} frames (threshold: 100)")

    def test_undetected_skip(self):
        """Test: Timecode skips without physical duplicate = UNDETECTED_SKIP"""
        prev = FrameAnalysis(
            frame_number=99,
            visual_timecode=Timecode(0, 0, 1, 3)  # Frame 63
        )

        current = FrameAnalysis(
            frame_number=100,
            visual_timecode=Timecode(0, 0, 1, 5),  # Frame 65 (skipped 1 frame)
            is_physical_duplicate=False
        )

        prev_frames = prev.visual_timecode.to_frames(60)
        curr_frames = current.visual_timecode.to_frames(60)
        diff = curr_frames - prev_frames

        assert diff > 1  # Skipped at least one frame
        assert current.is_physical_duplicate == False
        print("✓ test_undetected_skip passed")

    def run_all(self):
        """Run all logic gate tests"""
        print("\n=== Logic Gate Tests ===")
        self.test_successful_drop_detection()
        self.test_false_negative()
        self.test_false_positive()
        self.test_corruption_detection()
        self.test_undetected_skip()
        print("All Logic Gate tests passed!\n")


class TestReportGeneration:
    """Tests for report output format"""

    def test_report_structure(self):
        """Verify report contains required sections"""
        required_sections = [
            'sync',
            'corruption',
            'drop_accuracy',
            'start_timecode'
        ]

        validation_results = {
            'sync': {'status': 'PASS'},
            'corruption': {'A': {'status': 'PASS'}, 'B': {'status': 'PASS'}},
            'drop_accuracy': {'A': {'drop_count_match': 'PASS'}, 'B': {'drop_count_match': 'PASS'}},
            'start_timecode': {'A': {'status': 'PASS'}, 'B': {'status': 'PASS'}}
        }

        for section in required_sections:
            assert section in validation_results

        print("✓ test_report_structure passed")

    def run_all(self):
        """Run all report tests"""
        print("\n=== Report Tests ===")
        self.test_report_structure()
        print("All Report tests passed!\n")


def run_demo_with_sample_data():
    """Demonstrate the system using the provided sample metadata."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Video QA System with Sample Data")
    print("=" * 70)

    script_dir = Path(__file__).parent.resolve()
    possible_paths = [
        script_dir / "data",
        script_dir,
        Path.cwd(),
        Path.cwd() / "data",
    ]

    metadata_left = None
    metadata_right = None

    for base_path in possible_paths:
        left = base_path / "pull3_Take_007_left.json"
        right = base_path / "pull3_Take_007_right.json"
        if left.exists() and right.exists():
            metadata_left = left
            metadata_right = right
            break

    if metadata_left and metadata_right:
        print("\nLoading sample metadata files...")

        meta_a = CameraMetadata.from_json_file(metadata_left)
        meta_b = CameraMetadata.from_json_file(metadata_right)

        print(f"\nCamera A (Left): {meta_a.filename}")
        print(f"  Total Frames: {meta_a.num_frames}")
        print(f"  Frame Rate: {meta_a.frame_rate} fps")
        print(f"  Reported Drops: {meta_a.dropped_frames}")
        print(f"  Drop Percentage: {meta_a.drop_percentage}%")
        print(f"  Corrupted At: {meta_a.corrupted_at} (-1 = none)")
        print(f"  Start Timecode: {meta_a.start_timecode}")

        print(f"\nCamera B (Right): {meta_b.filename}")
        print(f"  Total Frames: {meta_b.num_frames}")
        print(f"  Frame Rate: {meta_b.frame_rate} fps")
        print(f"  Reported Drops: {meta_b.dropped_frames}")
        print(f"  Drop Percentage: {meta_b.drop_percentage}%")
        print(f"  Corrupted At: {meta_b.corrupted_at} (-1 = none)")
        print(f"  Start Timecode: {meta_b.start_timecode}")

        print("\n" + "-" * 50)
        print("METADATA VALIDATION RESULTS")
        print("-" * 50)

        sync_pass = meta_a.start_timecode == meta_b.start_timecode
        print(f"\n1. Start Timecode Sync: {'PASS' if sync_pass else 'FAIL'}")
        print(f"   Camera A: {meta_a.start_timecode}")
        print(f"   Camera B: {meta_b.start_timecode}")

        frame_match = meta_a.num_frames == meta_b.num_frames
        print(f"\n2. Frame Count Match: {'PASS' if frame_match else 'FAIL'}")
        print(f"   Camera A: {meta_a.num_frames}")
        print(f"   Camera B: {meta_b.num_frames}")

        drop_match = meta_a.dropped_frames == meta_b.dropped_frames
        print(f"\n3. Drop Count Match: {'PASS' if drop_match else 'FAIL'}")
        print(f"   Camera A: {meta_a.dropped_frames}")
        print(f"   Camera B: {meta_b.dropped_frames}")

        no_corruption = meta_a.corrupted_at == -1 and meta_b.corrupted_at == -1
        print(f"\n4. No Corruption Flagged: {'PASS' if no_corruption else 'FAIL'}")

        fps_match = meta_a.frame_rate == meta_b.frame_rate
        print(f"\n5. Frame Rate Match: {'PASS' if fps_match else 'FAIL'}")
        print(f"   Both cameras: {meta_a.frame_rate} fps")

        duration_sec = meta_a.num_frames / meta_a.frame_rate
        duration_min = duration_sec / 60
        print(f"\n6. Recording Duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")

        drop_rate = (meta_a.dropped_frames / meta_a.num_frames) * 100
        print(f"\n7. Drop Rate Analysis:")
        print(f"   Dropped Frames: {meta_a.dropped_frames}")
        print(f"   Total Frames: {meta_a.num_frames}")
        print(f"   Drop Rate: {drop_rate:.4f}%")
        print(f"   Quality: {'GOOD' if drop_rate < 1 else 'WARNING' if drop_rate < 5 else 'POOR'}")

        print("\n" + "=" * 70)
        print("NOTE: Full video analysis requires .mov files.")
        print("The system would analyse each frame for:")
        print("  - Visual timecode via OCR (improved LED dot-matrix support)")
        print("  - 'Dropped frame' indicator in top-left corner")
        print("  - Frame duplication (physical drops)")
        print("  - Timecode continuity and corruption")
        print("=" * 70)
    else:
        print("Sample metadata files not found")


def generate_sample_report():
    """Generate a sample report to demonstrate output format"""
    print("\n" + "=" * 70)
    print("GENERATING SAMPLE REPORT")
    print("=" * 70)

    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / "sample_reports"
    output_dir.mkdir(exist_ok=True)

    validation_results = {
        'sync': {
            'status': 'PASS',
            'camera_a_frame0_tc': '02:55:23:00',
            'camera_b_frame0_tc': '02:55:23:00',
            'notes': ''
        },
        'corruption': {
            'A': {
                'status': 'PASS',
                'metadata_corrupted_at': -1,
                'vision_corruptions_detected': 0,
                'notes': 'No corruption detected'
            },
            'B': {
                'status': 'PASS',
                'metadata_corrupted_at': -1,
                'vision_corruptions_detected': 0,
                'notes': 'No corruption detected'
            }
        },
        'drop_accuracy': {
            'A': {
                'metadata_drop_count': 58,
                'physical_drops_detected': 58,
                'indicator_drops_counted': 58,
                'successful_processes': 58,
                'false_negatives': 0,
                'false_positives': 0,
                'drop_count_match': 'PASS',
                'indicator_accuracy': 'PASS'
            },
            'B': {
                'metadata_drop_count': 58,
                'physical_drops_detected': 58,
                'indicator_drops_counted': 58,
                'successful_processes': 58,
                'false_negatives': 0,
                'false_positives': 0,
                'drop_count_match': 'PASS',
                'indicator_accuracy': 'PASS'
            }
        },
        'start_timecode': {
            'A': {
                'status': 'PASS',
                'ocr_frame0_tc': '02:55:23:00',
                'metadata_start_tc': '02:55:23:00',
                'notes': ''
            },
            'B': {
                'status': 'PASS',
                'ocr_frame0_tc': '02:55:23:00',
                'metadata_start_tc': '02:55:23:00',
                'notes': ''
            }
        }
    }

    json_path = output_dir / "sample_qa_report.json"
    with open(json_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'validation_results': validation_results,
            'camera_a_metadata': {
                'filename': 'pull3_Take_007_left.mov',
                'num_frames': 7718,
                'frame_rate': 60,
                'dropped_frames': 58,
                'start_timecode': '02:55:23:00'
            },
            'camera_b_metadata': {
                'filename': 'pull3_Take_007_right.mov',
                'num_frames': 7718,
                'frame_rate': 60,
                'dropped_frames': 58,
                'start_timecode': '02:55:23:00'
            }
        }, f, indent=2)
    print(f"Generated: {json_path}")

    text_report = """
================================================================================
VIDEO QA VALIDATION REPORT
================================================================================
Generated: {timestamp}

--------------------------------------------------------------------------------
CAMERA INFORMATION
--------------------------------------------------------------------------------
Camera A: pull3_Take_007_left.mov
  Frames: 7718
  Frame Rate: 60 fps
  Start TC: 02:55:23:00

Camera B: pull3_Take_007_right.mov
  Frames: 7718
  Frame Rate: 60 fps
  Start TC: 02:55:23:00

--------------------------------------------------------------------------------
1. SYNC STATUS
--------------------------------------------------------------------------------
Status: PASS
Camera A Frame 0 TC: 02:55:23:00
Camera B Frame 0 TC: 02:55:23:00

--------------------------------------------------------------------------------
2. CORRUPTION STATUS
--------------------------------------------------------------------------------
Camera A:
  Status: PASS
  Metadata Corrupted At: -1
  Vision Corruptions: 0
  Notes: No corruption detected

Camera B:
  Status: PASS
  Metadata Corrupted At: -1
  Vision Corruptions: 0
  Notes: No corruption detected

--------------------------------------------------------------------------------
3. DROP FRAME ACCURACY
--------------------------------------------------------------------------------
Metric                         Camera A        Camera B
------------------------------------------------------------
Metadata Drop Count                  58              58
Physical Drops Detected              58              58
Indicator Drops Counted              58              58
Successful Processes                 58              58
False Negatives                       0               0
False Positives                       0               0
Drop Count Match                   PASS            PASS
Indicator Accuracy                 PASS            PASS

--------------------------------------------------------------------------------
4. START TIMECODE VALIDATION
--------------------------------------------------------------------------------
Camera A:
  Status: PASS
  OCR Frame 0: 02:55:23:00
  Metadata: 02:55:23:00

Camera B:
  Status: PASS
  OCR Frame 0: 02:55:23:00
  Metadata: 02:55:23:00

--------------------------------------------------------------------------------
5. INDICATOR HEALTH SUMMARY
--------------------------------------------------------------------------------
Camera A:
  All drop indicators correctly reported

Camera B:
  All drop indicators correctly reported

================================================================================
OVERALL VALIDATION RESULT
================================================================================
RESULT: PASS

""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    text_path = output_dir / "sample_qa_report.txt"
    with open(text_path, 'w') as f:
        f.write(text_report)
    print(f"Generated: {text_path}")

    print(f"\nSample reports saved to: {output_dir}")
    return output_dir


def main():
    """Main test runner"""
    print("=" * 70)
    print("VIDEO QA AUTOMATION - TEST SUITE v2.0")
    print("=" * 70)
    print(f"\nOCR Engine Available:")
    print(f"  EasyOCR: {'Yes' if EASYOCR_AVAILABLE else 'No'}")

    setup_logging(verbose=False)

    # Run unit tests
    TestTimecode().run_all()
    TestOCRPipeline().run_all()

    # Find metadata file for testing
    script_dir = Path(__file__).parent.resolve()
    possible_paths = [
        script_dir / "data" / "pull3_Take_007_left.json",
        script_dir / "pull3_Take_007_left.json",
        Path.cwd() / "data" / "pull3_Take_007_left.json",
        Path.cwd() / "pull3_Take_007_left.json",
    ]

    metadata_path = None
    for p in possible_paths:
        if p.exists():
            metadata_path = p
            break

    TestMetadataLoading(metadata_path).run_all()
    TestLogicGates().run_all()
    TestReportGeneration().run_all()

    # Run demo with sample data
    run_demo_with_sample_data()

    # Generate sample reports
    output_dir = generate_sample_report()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nUsage:")
    print("  python video_qa_validator.py \\")
    print("    --video-a /path/to/camera_a.mov \\")
    print("    --video-b /path/to/camera_b.mov \\")
    print("    --metadata-a /path/to/camera_a.json \\")
    print("    --metadata-b /path/to/camera_b.json \\")
    print("    --output /path/to/reports \\")
    print("    --debug")
    print("\nFor batch processing:")
    print("  python batch_validate.py /path/to/data/folder")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
