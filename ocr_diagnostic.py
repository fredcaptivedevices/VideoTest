#!/usr/bin/env python3
"""
OCR Diagnostic Tool for Video QA Automation
============================================

This tool helps diagnose and tune OCR settings for LED timecode displays.
It extracts frames from video files and tests different OCR strategies,
saving debug images and results for analysis.

Usage:
    python ocr_diagnostic.py /path/to/video.mov
    python ocr_diagnostic.py /path/to/video.mov --frame 100
    python ocr_diagnostic.py /path/to/video.mov --roi 682,1390,679,257
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import argparse

# Try to import OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print(f"[✓] Tesseract available")
except ImportError:
    TESSERACT_AVAILABLE = False
    print(f"[✗] Tesseract NOT available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print(f"[✓] EasyOCR available")
except ImportError:
    EASYOCR_AVAILABLE = False
    print(f"[✗] EasyOCR NOT available")


# Global EasyOCR reader
_READER = None

def get_easyocr_reader():
    """Get or create EasyOCR reader"""
    global _READER
    if _READER is None and EASYOCR_AVAILABLE:
        print("Initializing EasyOCR (this may take a moment)...")
        try:
            import torch
            use_gpu = torch.cuda.is_available()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                use_gpu = True
            _READER = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            print(f"  EasyOCR initialized (GPU: {use_gpu})")
        except Exception as e:
            print(f"  Failed to initialize EasyOCR: {e}")
    return _READER


def load_frame(video_path: Path, frame_num: int = 0) -> Optional[np.ndarray]:
    """Load a specific frame from video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read frame {frame_num}")
        return None

    return frame


def extract_roi(frame: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    """Extract ROI from frame"""
    return frame[roi['y']:roi['y']+roi['height'], roi['x']:roi['x']+roi['width']]


def preprocess_strategies(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate multiple preprocessed versions optimized for LED dot-matrix displays.

    Output format: BLACK text on WHITE background (standard OCR convention).
    """
    results = []

    # Detect if light-on-dark or dark-on-light
    mean_val = np.mean(gray)
    is_light_on_dark = mean_val < 128

    # =========================================================================
    # Strategy 1: Heavy Gaussian blur to fuse LED dots
    # =========================================================================
    for blur_size in [9, 11, 15, 21]:
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        for thresh_val in [140, 160, 180, 200, 220]:
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

            # Morphological closing to connect segments
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # For OCR: black text on white background
            if is_light_on_dark:
                closed = cv2.bitwise_not(closed)

            results.append((f'blur{blur_size}_t{thresh_val}', closed))

    # =========================================================================
    # Strategy 2: Otsu's threshold (auto-level detection)
    # =========================================================================
    for blur_size in [7, 11, 15]:
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)

        # For OCR: black text on white background
        if is_light_on_dark:
            closed = cv2.bitwise_not(closed)

        results.append((f'otsu_blur{blur_size}', closed))

    # =========================================================================
    # Strategy 3: Adaptive threshold (handles uneven lighting)
    # =========================================================================
    for blur_size in [7, 11]:
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

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

    # =========================================================================
    # Strategy 4: Morphological reconstruction
    # =========================================================================
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    for thresh_val in [160, 180, 200]:
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Heavy dilation to connect dots
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)

        # Erode to restore shape
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(dilated, kernel_erode, iterations=1)

        # For OCR: black text on white background
        if is_light_on_dark:
            eroded = cv2.bitwise_not(eroded)

        results.append((f'morph_t{thresh_val}', eroded))

    # =========================================================================
    # Strategy 5: CLAHE contrast enhancement
    # =========================================================================
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

    # =========================================================================
    # Strategy 6: Bilateral filter (preserves edges)
    # =========================================================================
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    for thresh_val in [160, 180, 200]:
        _, thresh = cv2.threshold(bilateral, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # For OCR: black text on white background
        if is_light_on_dark:
            closed = cv2.bitwise_not(closed)

        results.append((f'bilateral_t{thresh_val}', closed))

    return results


def ocr_tesseract(img: np.ndarray, scale: float = 2.0) -> Tuple[str, float]:
    """Run Tesseract OCR"""
    if not TESSERACT_AVAILABLE:
        return "", 0.0

    try:
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Multiple PSM modes to try
        configs = [
            '--psm 7 -c tessedit_char_whitelist=0123456789:',  # Single line
            '--psm 8 -c tessedit_char_whitelist=0123456789:',  # Single word
            '--psm 13 -c tessedit_char_whitelist=0123456789:', # Raw line
            '--psm 6 -c tessedit_char_whitelist=0123456789:',  # Single block
        ]

        best_text = ""
        best_conf = 0.0

        for config in configs:
            try:
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

                texts = []
                confidences = []
                for i, text in enumerate(data['text']):
                    text = text.strip()
                    conf = data['conf'][i]
                    if text and conf > 0:
                        texts.append(text)
                        confidences.append(conf)

                if texts:
                    combined = ''.join(texts)
                    avg_conf = sum(confidences) / len(confidences)

                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_text = combined
            except:
                pass

        return best_text, best_conf
    except Exception as e:
        return f"Error: {e}", 0.0


def ocr_easyocr(img: np.ndarray, scale: float = 2.0) -> Tuple[str, float]:
    """Run EasyOCR with various settings"""
    reader = get_easyocr_reader()
    if reader is None:
        return "", 0.0

    try:
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to BGR for EasyOCR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Try different EasyOCR configurations
        configs = [
            # Standard config
            {'allowlist': '0123456789:', 'paragraph': False, 'detail': 1,
             'width_ths': 0.5, 'height_ths': 0.5},
            # More permissive
            {'allowlist': '0123456789:', 'paragraph': False, 'detail': 1,
             'width_ths': 0.3, 'height_ths': 0.3, 'contrast_ths': 0.1},
            # Tight boxes
            {'allowlist': '0123456789:', 'paragraph': True, 'detail': 1,
             'width_ths': 0.7, 'height_ths': 0.7},
            # Low text threshold
            {'allowlist': '0123456789:', 'paragraph': False, 'detail': 1,
             'low_text': 0.3, 'text_threshold': 0.5},
        ]

        best_text = ""
        best_conf = 0.0

        for config in configs:
            try:
                results = reader.readtext(img, **config)

                if results:
                    # Sort by x-position
                    sorted_results = sorted(results, key=lambda r: r[0][0][0])

                    texts = []
                    confidences = []
                    for (bbox, text, conf) in sorted_results:
                        texts.append(text.strip())
                        confidences.append(conf)

                    combined = ''.join(texts)
                    avg_conf = sum(confidences) / len(confidences) * 100

                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_text = combined
            except:
                pass

        return best_text, best_conf
    except Exception as e:
        return f"Error: {e}", 0.0


def clean_ocr_text(text: str) -> str:
    """Clean up OCR text"""
    replacements = {
        '.': ':', ' ': '', 'O': '0', 'o': '0',
        'l': '1', 'I': '1', '|': '1', 'S': '5',
        'B': '8', 'Z': '2', 'G': '6', 'D': '0'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def parse_timecode(text: str) -> Optional[str]:
    """Try to parse timecode from OCR text"""
    import re

    text = clean_ocr_text(text)

    # Standard pattern: HH:MM:SS:FF
    match = re.search(r'(\d{1,2}):(\d{2}):(\d{2}):(\d{2})', text)
    if match:
        h, m, s, f = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        if 0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59 and 0 <= f <= 59:
            return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    # Loose pattern: digits with any separator
    match = re.search(r'(\d{1,2})\D*(\d{2})\D*(\d{2})\D*(\d{2})', text)
    if match:
        h, m, s, f = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        if 0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59 and 0 <= f <= 59:
            return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    # 8-digit pattern: HHMMSSFF
    digits = re.sub(r'\D', '', text)
    if len(digits) == 8:
        h, m, s, f = int(digits[0:2]), int(digits[2:4]), int(digits[4:6]), int(digits[6:8])
        if 0 <= h <= 99 and 0 <= m <= 59 and 0 <= s <= 59 and 0 <= f <= 59:
            return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    return None


def run_diagnostic(video_path: Path, frame_num: int, roi: Optional[Dict[str, int]],
                   output_dir: Path, expected_tc: Optional[str] = None):
    """Run full diagnostic on a video frame"""

    print(f"\n{'='*70}")
    print(f"OCR DIAGNOSTIC TOOL")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Frame: {frame_num}")
    if expected_tc:
        print(f"Expected TC: {expected_tc}")

    # Load frame
    frame = load_frame(video_path, frame_num)
    if frame is None:
        return

    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save original frame
    cv2.imwrite(str(output_dir / f"frame_{frame_num:04d}_original.png"), frame)

    # Auto-detect or use provided ROI
    if roi is None:
        print("\nNo ROI provided - attempting auto-detection...")
        roi = auto_detect_roi(frame)
        if roi:
            print(f"Auto-detected ROI: {roi}")
        else:
            print("Auto-detection failed. Using center region.")
            h, w = frame.shape[:2]
            roi = {'x': w//4, 'y': h//3, 'width': w//2, 'height': h//3}

    print(f"ROI: x={roi['x']}, y={roi['y']}, w={roi['width']}, h={roi['height']}")

    # Extract ROI
    roi_img = extract_roi(frame, roi)
    cv2.imwrite(str(output_dir / f"frame_{frame_num:04d}_roi.png"), roi_img)

    # Save frame with ROI marked
    marked = frame.copy()
    cv2.rectangle(marked, (roi['x'], roi['y']),
                  (roi['x']+roi['width'], roi['y']+roi['height']), (0, 255, 0), 3)
    cv2.imwrite(str(output_dir / f"frame_{frame_num:04d}_marked.png"), marked)

    # Convert to grayscale
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    cv2.imwrite(str(output_dir / f"frame_{frame_num:04d}_gray.png"), gray)

    # Generate preprocessed versions
    print(f"\nGenerating preprocessed images...")
    preprocessed = preprocess_strategies(gray)
    print(f"  Generated {len(preprocessed)} preprocessing variants")

    # Save first 10 preprocessed images
    for i, (name, img) in enumerate(preprocessed[:10]):
        cv2.imwrite(str(output_dir / f"frame_{frame_num:04d}_pp{i:02d}_{name}.png"), img)

    # Run OCR on all variants
    print(f"\nRunning OCR tests...")
    results = []

    for name, img in preprocessed:
        for scale in [2.0, 3.0]:
            # Tesseract
            if TESSERACT_AVAILABLE:
                text, conf = ocr_tesseract(img, scale)
                parsed = parse_timecode(text)
                results.append({
                    'method': 'tesseract',
                    'preprocess': name,
                    'scale': scale,
                    'raw': text,
                    'parsed': parsed,
                    'confidence': conf,
                    'valid': parsed is not None,
                    'correct': parsed == expected_tc if expected_tc else None
                })

            # EasyOCR
            if EASYOCR_AVAILABLE:
                text, conf = ocr_easyocr(img, scale)
                parsed = parse_timecode(text)
                results.append({
                    'method': 'easyocr',
                    'preprocess': name,
                    'scale': scale,
                    'raw': text,
                    'parsed': parsed,
                    'confidence': conf,
                    'valid': parsed is not None,
                    'correct': parsed == expected_tc if expected_tc else None
                })

    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Print results
    print(f"\n{'='*70}")
    print(f"OCR RESULTS (Top 20 by confidence)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Preprocess':<25} {'Scale':<6} {'Raw':<20} {'Parsed':<15} {'Conf':<8} {'Valid'}")
    print("-" * 100)

    for r in results[:20]:
        valid_str = "✓" if r['valid'] else "✗"
        if r['correct'] is True:
            valid_str = "✓✓"
        elif r['correct'] is False:
            valid_str = "✗!"

        print(f"{r['method']:<12} {r['preprocess'][:24]:<25} {r['scale']:<6.1f} "
              f"{r['raw'][:19]:<20} {str(r['parsed']):<15} {r['confidence']:<8.1f} {valid_str}")

    # Summary
    valid_count = sum(1 for r in results if r['valid'])
    correct_count = sum(1 for r in results if r['correct'] is True)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total OCR attempts: {len(results)}")
    print(f"Valid timecodes found: {valid_count} ({100*valid_count/len(results):.1f}%)")
    if expected_tc:
        print(f"Correct timecodes: {correct_count} ({100*correct_count/len(results):.1f}%)")

    # Best result
    valid_results = [r for r in results if r['valid']]
    if valid_results:
        best = valid_results[0]
        print(f"\nBest result:")
        print(f"  Method: {best['method']}")
        print(f"  Preprocessing: {best['preprocess']}")
        print(f"  Scale: {best['scale']}")
        print(f"  Timecode: {best['parsed']}")
        print(f"  Confidence: {best['confidence']:.1f}")
    else:
        print(f"\nNo valid timecodes found!")

    # Save results to JSON
    json_path = output_dir / f"frame_{frame_num:04d}_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'frame_num': frame_num,
            'roi': roi,
            'expected_tc': expected_tc,
            'results': results[:50],  # Top 50
            'summary': {
                'total_attempts': len(results),
                'valid_count': valid_count,
                'correct_count': correct_count if expected_tc else None
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


def auto_detect_roi(frame: np.ndarray) -> Optional[Dict[str, int]]:
    """Attempt to auto-detect timecode ROI"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Look for bright regions (LED display)
    for thresh_val in [220, 200, 180]:
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # Dilate to connect LED segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
        dilated = cv2.dilate(mask, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:10]:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0

            # Timecode display: wide rectangle
            if 2.5 < aspect < 10.0 and cw > 80 and ch > 15:
                padding = 50
                return {
                    'x': max(0, x - padding),
                    'y': max(0, y - padding),
                    'width': min(cw + 2*padding, w - x + padding),
                    'height': min(ch + 2*padding, h - y + padding)
                }

    return None


def main():
    parser = argparse.ArgumentParser(
        description='OCR Diagnostic Tool for Video QA',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--frame', '-f', type=int, default=0, help='Frame number to analyze')
    parser.add_argument('--roi', '-r', help='ROI as x,y,width,height (e.g., 682,1390,679,257)')
    parser.add_argument('--expected', '-e', help='Expected timecode (e.g., 02:55:23:00)')
    parser.add_argument('--output', '-o', default='ocr_diagnostic_output', help='Output directory')
    parser.add_argument('--multi', '-m', type=int, help='Analyze multiple frames (every N frames)')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return 1

    # Parse ROI
    roi = None
    if args.roi:
        try:
            parts = [int(x) for x in args.roi.split(',')]
            roi = {'x': parts[0], 'y': parts[1], 'width': parts[2], 'height': parts[3]}
        except:
            print(f"Error: Invalid ROI format. Use: x,y,width,height")
            return 1

    output_dir = Path(args.output)

    if args.multi:
        # Multi-frame analysis
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        for frame_num in range(0, total_frames, args.multi):
            run_diagnostic(video_path, frame_num, roi, output_dir, args.expected)
    else:
        run_diagnostic(video_path, args.frame, roi, output_dir, args.expected)

    return 0


if __name__ == '__main__':
    sys.exit(main())
