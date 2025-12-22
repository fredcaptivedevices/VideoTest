#!/usr/bin/env python3
"""
ROI Calibration GUI for Video QA Automation
============================================

Allows manual selection of timecode ROI by dragging a box around the timecode display.
The ROI is saved and reused for both cameras across all takes in a shot.

Usage:
    python calibrate_roi.py /path/to/video.mov
    python calibrate_roi.py /path/to/take/folder
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple


class ROISelector:
    """
    Interactive GUI for selecting the timecode ROI.
    Uses OpenCV's mouse callbacks for click-and-drag selection.
    """
    
    def __init__(self, frame: np.ndarray, window_name: str = "Select Timecode ROI"):
        self.original_frame = frame.copy()
        self.display_frame = frame.copy()
        self.window_name = window_name
        
        # ROI coordinates
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.drawing = False
        self.roi_selected = False
        
        # For display scaling (large frames need to fit on screen)
        self.scale = 1.0
        self.display_height = frame.shape[0]
        self.display_width = frame.shape[1]
        
        # Scale down if frame is too large
        max_display_height = 900
        max_display_width = 1200
        
        if frame.shape[0] > max_display_height or frame.shape[1] > max_display_width:
            scale_h = max_display_height / frame.shape[0]
            scale_w = max_display_width / frame.shape[1]
            self.scale = min(scale_h, scale_w)
            self.display_height = int(frame.shape[0] * self.scale)
            self.display_width = int(frame.shape[1] * self.scale)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        # Convert display coordinates to original frame coordinates
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (orig_x, orig_y)
            self.end_point = (orig_x, orig_y)
            self.roi_selected = False
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (orig_x, orig_y)
            self._update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (orig_x, orig_y)
            self.roi_selected = True
            self._update_display()
    
    def _update_display(self):
        """Redraw the frame with current ROI rectangle"""
        self.display_frame = self.original_frame.copy()
        
        if self.start_point and self.end_point:
            # Draw rectangle
            cv2.rectangle(
                self.display_frame,
                self.start_point,
                self.end_point,
                (0, 255, 0),  # Green
                3
            )
            
            # Draw corner handles
            for point in [self.start_point, self.end_point,
                         (self.start_point[0], self.end_point[1]),
                         (self.end_point[0], self.start_point[1])]:
                cv2.circle(self.display_frame, point, 8, (0, 255, 0), -1)
    
    def get_roi(self) -> Optional[Dict[str, int]]:
        """
        Display the GUI and return the selected ROI.
        Returns None if cancelled.
        """
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("ROI CALIBRATION")
        print("=" * 60)
        print("Instructions:")
        print("  1. Click and drag to draw a box around the TIMECODE DISPLAY")
        print("  2. Make sure the box includes the full timecode (00:00:00:00)")
        print("  3. Press ENTER or SPACE to confirm")
        print("  4. Press 'R' to reset and try again")
        print("  5. Press ESC or 'Q' to cancel")
        print("=" * 60 + "\n")
        
        while True:
            # Scale for display
            if self.scale != 1.0:
                display = cv2.resize(self.display_frame, (self.display_width, self.display_height))
            else:
                display = self.display_frame.copy()
            
            # Add instructions overlay
            cv2.putText(display, "Draw box around timecode, then press ENTER to confirm",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Press R to reset, ESC to cancel",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show ROI info if selected
            if self.roi_selected and self.start_point and self.end_point:
                roi = self._calculate_roi()
                cv2.putText(display, f"ROI: x={roi['x']}, y={roi['y']}, w={roi['width']}, h={roi['height']}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Enter or Space to confirm
            if key in [13, 32] and self.roi_selected:
                cv2.destroyAllWindows()
                return self._calculate_roi()
            
            # R to reset
            elif key == ord('r') or key == ord('R'):
                self.start_point = None
                self.end_point = None
                self.roi_selected = False
                self.display_frame = self.original_frame.copy()
            
            # ESC or Q to cancel
            elif key == 27 or key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return None
            
            # Window closed
            elif cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def _calculate_roi(self) -> Dict[str, int]:
        """Calculate ROI dict from start/end points"""
        if not self.start_point or not self.end_point:
            return {}
        
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Normalise to ensure x1,y1 is top-left
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        
        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        return {
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }


def load_first_frame(video_path: Path) -> Optional[np.ndarray]:
    """Load the first frame from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read first frame from {video_path}")
        return None
    
    return frame


def find_video_in_folder(folder_path: Path) -> Optional[Path]:
    """Find a video file in a take folder"""
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv']
    
    for ext in video_extensions:
        videos = list(folder_path.glob(f'*{ext}'))
        if videos:
            # Prefer left camera
            for v in videos:
                if '_left' in v.name.lower():
                    return v
            return videos[0]
    
    return None


def save_roi_config(roi: Dict[str, int], output_path: Path):
    """Save ROI configuration to JSON file"""
    config = {
        'timecode_roi': roi,
        'version': '1.0',
        'note': 'Generated by calibrate_roi.py - applies to all cameras in shot'
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"ROI saved to: {output_path}")


def load_roi_config(config_path: Path) -> Optional[Dict[str, int]]:
    """Load ROI configuration from JSON file"""
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('timecode_roi')
    except Exception as e:
        print(f"Error loading ROI config: {e}")
        return None


def calibrate_from_video(video_path: Path) -> Optional[Dict[str, int]]:
    """Run calibration GUI on a video file"""
    print(f"Loading video: {video_path}")
    
    frame = load_first_frame(video_path)
    if frame is None:
        return None
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    selector = ROISelector(frame, f"Select Timecode ROI - {video_path.name}")
    roi = selector.get_roi()
    
    return roi


def calibrate_take_folder(take_path: Path) -> Optional[Dict[str, int]]:
    """Run calibration for a take folder and save config"""
    video = find_video_in_folder(take_path)
    if not video:
        print(f"Error: No video files found in {take_path}")
        return None
    
    roi = calibrate_from_video(video)
    
    if roi:
        # Save to take folder
        config_path = take_path / 'roi_config.json'
        save_roi_config(roi, config_path)
        
        # Also save to parent (shot) folder for reuse
        shot_folder = take_path.parent
        shot_config_path = shot_folder / 'roi_config.json'
        save_roi_config(roi, shot_config_path)
        print(f"ROI also saved to shot folder: {shot_config_path}")
    
    return roi


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python calibrate_roi.py /path/to/video.mov")
        print("  python calibrate_roi.py /path/to/take/folder")
        print("")
        print("The ROI will be saved as roi_config.json and used for all cameras.")
        return 1
    
    input_path = Path(sys.argv[1]).resolve()
    
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        return 1
    
    if input_path.is_file():
        # Direct video file
        roi = calibrate_from_video(input_path)
        if roi:
            config_path = input_path.parent / 'roi_config.json'
            save_roi_config(roi, config_path)
            print(f"\nROI Configuration:")
            print(f"  x: {roi['x']}")
            print(f"  y: {roi['y']}")
            print(f"  width: {roi['width']}")
            print(f"  height: {roi['height']}")
        else:
            print("Calibration cancelled.")
            return 1
    
    elif input_path.is_dir():
        # Take folder
        roi = calibrate_take_folder(input_path)
        if roi:
            print(f"\nROI Configuration:")
            print(f"  x: {roi['x']}")
            print(f"  y: {roi['y']}")
            print(f"  width: {roi['width']}")
            print(f"  height: {roi['height']}")
        else:
            print("Calibration cancelled.")
            return 1
    
    print("\nCalibration complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
