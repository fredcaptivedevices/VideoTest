#!/usr/bin/env python3
"""
Batch Video QA Processor
========================
Automatically discovers and validates all takes within a data directory.

Usage:
    python batch_validate.py                    # Process all takes in ./data/
    python batch_validate.py /path/to/data      # Process all takes in specified directory
    python batch_validate.py --take /path/to/specific/take  # Process single take
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import json

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from video_qa_validator import (
    DualCameraValidator,
    CameraMetadata,
    setup_logging
)


class TakeDiscovery:
    """Discovers and validates take folder structures"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path).resolve()
        self.logger = logging.getLogger("TakeDiscovery")
    
    def find_all_takes(self) -> List[Path]:
        """
        Recursively find all valid take folders.
        A valid take folder contains exactly 2 .mov files and 2 .json files
        with matching _left/_right naming.
        """
        takes = []
        
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            
            # Check if this folder looks like a take
            mov_files = [f for f in files if f.endswith('.mov')]
            json_files = [f for f in files if f.endswith('.json')]
            
            if len(mov_files) == 2 and len(json_files) == 2:
                # Verify we have left/right pairs
                if self._has_valid_pairs(mov_files, json_files):
                    takes.append(root_path)
                    self.logger.debug(f"Found valid take: {root_path}")
        
        # Sort by path for consistent ordering
        takes.sort()
        return takes
    
    def _has_valid_pairs(self, mov_files: List[str], json_files: List[str]) -> bool:
        """Check if files form valid left/right pairs"""
        mov_bases = {f.replace('.mov', '') for f in mov_files}
        json_bases = {f.replace('.json', '') for f in json_files}
        
        # Check that mov and json bases match
        if mov_bases != json_bases:
            return False
        
        # Check for left/right naming
        bases = list(mov_bases)
        has_left = any('_left' in b or '_Left' in b for b in bases)
        has_right = any('_right' in b or '_Right' in b for b in bases)
        
        return has_left and has_right
    
    def get_take_files(self, take_path: Path) -> Optional[Dict[str, Path]]:
        """
        Get the file paths for a take folder.
        Returns dict with keys: video_a, video_b, metadata_a, metadata_b
        """
        take_path = Path(take_path)
        
        mov_files = list(take_path.glob('*.mov'))
        json_files = list(take_path.glob('*.json'))
        
        if len(mov_files) != 2 or len(json_files) != 2:
            self.logger.error(f"Invalid take folder: {take_path}")
            return None
        
        # Sort to get consistent left/right assignment
        # Left = Camera A, Right = Camera B
        mov_files.sort(key=lambda p: p.name)
        json_files.sort(key=lambda p: p.name)
        
        # Identify left/right
        video_left = video_right = None
        meta_left = meta_right = None
        
        for mov in mov_files:
            name_lower = mov.name.lower()
            if '_left' in name_lower:
                video_left = mov
            elif '_right' in name_lower:
                video_right = mov
        
        for js in json_files:
            name_lower = js.name.lower()
            if '_left' in name_lower:
                meta_left = js
            elif '_right' in name_lower:
                meta_right = js
        
        if not all([video_left, video_right, meta_left, meta_right]):
            self.logger.error(f"Could not identify left/right pairs in: {take_path}")
            return None
        
        return {
            'video_a': video_left,
            'video_b': video_right,
            'metadata_a': meta_left,
            'metadata_b': meta_right,
            'take_name': take_path.name,
            'take_path': take_path
        }


class BatchValidator:
    """Processes multiple takes and generates summary reports"""
    
    def __init__(self, data_dir: Path, output_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "qa_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.discovery = TakeDiscovery(self.data_dir)
        self.logger = logging.getLogger("BatchValidator")
        
        self.results: List[Dict] = []
    
    def validate_single_take(self, take_path: Path, debug: bool = False) -> Optional[Dict]:
        """Validate a single take and return results"""
        files = self.discovery.get_take_files(take_path)
        if not files:
            return None
        
        take_name = files['take_name']
        self.logger.info(f"Processing: {take_name}")
        
        # Create take-specific output directory
        take_output = take_path / "qa_reports"
        take_output.mkdir(exist_ok=True)
        
        try:
            validator = DualCameraValidator(
                video_a=files['video_a'],
                video_b=files['video_b'],
                metadata_a=files['metadata_a'],
                metadata_b=files['metadata_b'],
                output_dir=take_output
            )
            
            # Show which files we're processing
            print(f"\n  Camera A: {files['video_a'].name}")
            print(f"  Camera B: {files['video_b'].name}")
            
            success = validator.run_full_validation()
            
            if success:
                text_path, html_path = validator.generate_report()
                
                result = {
                    'take_name': take_name,
                    'take_path': str(take_path),
                    'status': 'COMPLETED',
                    'validation_results': validator.validation_results,
                    'camera_a_stats': validator.analyser_a.stats,
                    'camera_b_stats': validator.analyser_b.stats,
                    'report_html': html_path,
                    'report_text': text_path,
                    'overall_pass': self._check_overall_pass(validator.validation_results)
                }
                
                # Print quick result summary
                status_icon = "✓" if result['overall_pass'] else "✗"
                drops_a = validator.analyser_a.stats.get('physical_drops', 0)
                drops_b = validator.analyser_b.stats.get('physical_drops', 0)
                print(f"\n  Result: {status_icon} {'PASS' if result['overall_pass'] else 'FAIL'} | Drops: A={drops_a}, B={drops_b}")
            else:
                result = {
                    'take_name': take_name,
                    'take_path': str(take_path),
                    'status': 'FAILED',
                    'error': 'Validation failed to complete',
                    'overall_pass': False
                }
                print(f"\n  Result: ✗ FAILED - Validation did not complete")
                
        except Exception as e:
            self.logger.error(f"Error processing {take_name}: {e}")
            result = {
                'take_name': take_name,
                'take_path': str(take_path),
                'status': 'ERROR',
                'error': str(e),
                'overall_pass': False
            }
            print(f"\n  Result: ✗ ERROR - {e}")
        
        return result
    
    def _find_roi_config(self, take_path: Path) -> Optional[Dict[str, int]]:
        """Look for roi_config.json in take folder or parent folders"""
        search_paths = [
            take_path / 'roi_config.json',
            take_path.parent / 'roi_config.json',
            take_path.parent.parent / 'roi_config.json',
        ]
        
        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    roi = config.get('timecode_roi')
                    if roi:
                        return roi
                except Exception:
                    pass
        
        return None
    
    def _check_overall_pass(self, validation_results: Dict) -> bool:
        """Determine if all validation checks passed"""
        try:
            sync = validation_results.get('sync', {})
            corruption = validation_results.get('corruption', {})
            drop_data = validation_results.get('drop_accuracy', {})
            start_tc = validation_results.get('start_timecode', {})
            
            return all([
                sync.get('status') == 'PASS',
                all(c.get('status') == 'PASS' for c in corruption.values()),
                all(d.get('drop_count_match') == 'PASS' for d in drop_data.values()),
                all(d.get('indicator_accuracy') == 'PASS' for d in drop_data.values()),
                all(s.get('status') == 'PASS' for s in start_tc.values())
            ])
        except Exception:
            return False
    
    def validate_all_takes(self, debug: bool = False) -> List[Dict]:
        """Discover and validate all takes in the data directory"""
        takes = self.discovery.find_all_takes()
        total_takes = len(takes)
        
        if total_takes == 0:
            print(f"\nNo takes found in {self.data_dir}")
            return []
        
        print(f"\n{'='*60}")
        print(f"Found {total_takes} takes to process")
        print(f"{'='*60}")
        
        # Group takes by shot (parent folder)
        shots = {}
        for take_path in takes:
            shot_name = take_path.parent.name
            if shot_name not in shots:
                shots[shot_name] = []
            shots[shot_name].append(take_path)
        
        print(f"Organised into {len(shots)} shots")
        
        # Process each shot
        for shot_name, shot_takes in shots.items():
            print(f"\n{'='*60}")
            print(f"SHOT: {shot_name} ({len(shot_takes)} takes)")
            print(f"{'='*60}")
            
            # Check/create ROI config for this shot using first take
            shot_folder = shot_takes[0].parent
            roi_config = self._ensure_roi_config_for_shot(shot_folder, shot_takes[0])
            
            if roi_config is None:
                print(f"\n  Skipping shot {shot_name} - calibration cancelled")
                for take_path in shot_takes:
                    self.results.append({
                        'take_name': take_path.name,
                        'take_path': str(take_path),
                        'status': 'SKIPPED',
                        'error': 'ROI calibration cancelled',
                        'overall_pass': False
                    })
                continue
            
            # Process all takes in this shot
            for i, take_path in enumerate(shot_takes, 1):
                rel_path = take_path.relative_to(self.data_dir)
                
                print(f"\n{'─'*60}")
                print(f"[{i}/{len(shot_takes)}] {rel_path}")
                print(f"{'─'*60}")
                
                result = self.validate_single_take(take_path, debug=debug)
                if result:
                    self.results.append(result)
        
        # Print final summary
        passed = sum(1 for r in self.results if r.get('overall_pass', False))
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {passed}/{len(self.results)} takes passed")
        print(f"{'='*60}\n")
        
        return self.results
    
    def _ensure_roi_config_for_shot(self, shot_folder: Path, first_take: Path) -> Optional[Dict[str, int]]:
        """
        Always open calibration GUI for each shot.
        The ROI is saved for reference but recalibration is required each run.
        """
        print(f"\n  Opening calibration tool...")
        print(f"  → Draw a box around the TIMECODE DISPLAY")
        print(f"  → Press ENTER to confirm, ESC to skip this shot")
        
        # Find a video to calibrate with
        files = self.discovery.get_take_files(first_take)
        if not files:
            print(f"  Error: Could not find video files")
            return None
        
        video_path = files['video_a']  # Use left camera
        
        # Run calibration GUI
        try:
            from calibrate_roi import calibrate_from_video, save_roi_config
            
            roi_config = calibrate_from_video(video_path)
            
            if roi_config:
                # Save to shot folder (for reference/logging)
                config_path = shot_folder / 'roi_config.json'
                save_roi_config(roi_config, config_path)
                print(f"\n  ✓ ROI configured")
                print(f"  ✓ Will be used for all takes in this shot")
                return roi_config
            else:
                print(f"\n  ✗ Calibration cancelled")
                return None
                
        except ImportError as e:
            print(f"  Error: Could not import calibrate_roi module: {e}")
            print(f"  Make sure calibrate_roi.py is in the same directory")
            return None
        except Exception as e:
            print(f"  Error during calibration: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_batch_summary(self) -> Tuple[str, str]:
        """Generate summary report for all processed takes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Count statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('overall_pass', False))
        failed = sum(1 for r in self.results if r.get('status') == 'COMPLETED' and not r.get('overall_pass', False))
        errors = sum(1 for r in self.results if r.get('status') in ['FAILED', 'ERROR'])
        
        # Generate text summary
        text_lines = [
            "=" * 80,
            "BATCH VIDEO QA VALIDATION SUMMARY",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Directory: {self.data_dir}",
            "",
            "-" * 80,
            "SUMMARY",
            "-" * 80,
            f"Total Takes Processed: {total}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Errors: {errors}",
            f"Pass Rate: {(passed/total*100) if total > 0 else 0:.1f}%",
            "",
            "-" * 80,
            "RESULTS BY TAKE",
            "-" * 80,
        ]
        
        for result in self.results:
            status_icon = "✓" if result.get('overall_pass') else "✗"
            text_lines.append(f"{status_icon} {result['take_name']}: {result['status']}")
            
            if result.get('error'):
                text_lines.append(f"    Error: {result['error']}")
            elif result.get('validation_results'):
                vr = result['validation_results']
                text_lines.append(f"    Sync: {vr.get('sync', {}).get('status', 'N/A')}")
                
                # Show drop counts
                drop_a = result.get('camera_a_stats', {})
                drop_b = result.get('camera_b_stats', {})
                text_lines.append(f"    Drops - A: {drop_a.get('physical_drops', 'N/A')}, B: {drop_b.get('physical_drops', 'N/A')}")
                text_lines.append(f"    False Negatives - A: {drop_a.get('false_negatives', 'N/A')}, B: {drop_b.get('false_negatives', 'N/A')}")
            
            text_lines.append("")
        
        text_lines.extend([
            "=" * 80,
            f"OVERALL: {'ALL PASSED' if passed == total and total > 0 else 'ISSUES FOUND'}",
            "=" * 80,
        ])
        
        # Save text report
        text_path = self.output_dir / f"batch_summary_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write('\n'.join(text_lines))
        
        # Save JSON report
        json_path = self.output_dir / f"batch_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'data_directory': str(self.data_dir),
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'errors': errors,
                    'pass_rate': (passed/total*100) if total > 0 else 0
                },
                'results': self.results
            }, f, indent=2, default=str)
        
        # Generate HTML summary
        html_path = self.output_dir / f"batch_summary_{timestamp}.html"
        self._generate_html_summary(html_path, total, passed, failed, errors)
        
        self.logger.info(f"Batch summary saved to: {self.output_dir}")
        
        return str(text_path), str(html_path)
    
    def _generate_html_summary(self, html_path: Path, total: int, passed: int, failed: int, errors: int):
        """Generate HTML batch summary"""
        pass_rate = (passed/total*100) if total > 0 else 0
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch QA Summary</title>
    <style>
        :root {{
            --pass-color: #28a745;
            --fail-color: #dc3545;
            --warning-color: #ffc107;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        h1 {{ text-align: center; color: #343a40; }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card-value {{
            font-size: 48px;
            font-weight: bold;
        }}
        .card-label {{
            color: #6c757d;
            margin-top: 5px;
        }}
        .pass {{ color: var(--pass-color); }}
        .fail {{ color: var(--fail-color); }}
        .warning {{ color: var(--warning-color); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{ background-color: #343a40; color: white; }}
        tr:hover {{ background-color: #f1f3f4; }}
        .status-pass {{ color: var(--pass-color); font-weight: bold; }}
        .status-fail {{ color: var(--fail-color); font-weight: bold; }}
        a {{ color: #007bff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .timestamp {{ text-align: center; color: #6c757d; }}
    </style>
</head>
<body>
    <h1>Batch Video QA Validation Summary</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p class="timestamp">Data Directory: {self.data_dir}</p>
    
    <div class="summary-cards">
        <div class="card">
            <div class="card-value">{total}</div>
            <div class="card-label">Total Takes</div>
        </div>
        <div class="card">
            <div class="card-value pass">{passed}</div>
            <div class="card-label">Passed</div>
        </div>
        <div class="card">
            <div class="card-value fail">{failed + errors}</div>
            <div class="card-label">Failed/Errors</div>
        </div>
        <div class="card">
            <div class="card-value {'pass' if pass_rate >= 90 else 'warning' if pass_rate >= 70 else 'fail'}">{pass_rate:.1f}%</div>
            <div class="card-label">Pass Rate</div>
        </div>
    </div>
    
    <table>
        <thead>
            <tr>
                <th>Take</th>
                <th>Status</th>
                <th>Sync</th>
                <th>Drops (A/B)</th>
                <th>False Neg (A/B)</th>
                <th>Report</th>
            </tr>
        </thead>
        <tbody>
'''
        
        for result in self.results:
            status_class = "status-pass" if result.get('overall_pass') else "status-fail"
            status_text = "PASS" if result.get('overall_pass') else result.get('status', 'FAIL')
            
            vr = result.get('validation_results', {})
            sync_status = vr.get('sync', {}).get('status', 'N/A')
            
            stats_a = result.get('camera_a_stats', {})
            stats_b = result.get('camera_b_stats', {})
            drops = f"{stats_a.get('physical_drops', 'N/A')} / {stats_b.get('physical_drops', 'N/A')}"
            false_neg = f"{stats_a.get('false_negatives', 'N/A')} / {stats_b.get('false_negatives', 'N/A')}"
            
            report_link = ""
            if result.get('report_html'):
                report_link = f'<a href="file://{result["report_html"]}">View</a>'
            
            html += f'''
            <tr>
                <td>{result['take_name']}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{sync_status}</td>
                <td>{drops}</td>
                <td>{false_neg}</td>
                <td>{report_link}</td>
            </tr>
'''
        
        html += '''
        </tbody>
    </table>
</body>
</html>
'''
        
        with open(html_path, 'w') as f:
            f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='Batch Video QA Validation - Process multiple takes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all takes in ./data/
    python batch_validate.py
    
    # Process all takes in a specific directory
    python batch_validate.py /Volumes/Shoots/Project1/data
    
    # Process a single take
    python batch_validate.py --take /path/to/pull3/Take_007
    
    # List all discovered takes without processing
    python batch_validate.py --list-only
        """
    )
    
    parser.add_argument(
        'data_dir',
        nargs='?',
        default='./data',
        help='Base directory containing take folders (default: ./data)'
    )
    
    parser.add_argument(
        '--take', '-t',
        help='Process a single take folder instead of batch'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for batch summary (default: <data_dir>/qa_reports)'
    )
    
    parser.add_argument(
        '--list-only', '-l',
        action='store_true',
        help='List discovered takes without processing'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Save debug frames for flagged issues'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('main')
    
    # Resolve data directory
    data_dir = Path(args.data_dir).resolve()
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Single take mode
    if args.take:
        take_path = Path(args.take).resolve()
        if not take_path.exists():
            logger.error(f"Take folder not found: {take_path}")
            return 1
        
        logger.info(f"Processing single take: {take_path}")
        
        batch = BatchValidator(take_path.parent, args.output)
        result = batch.validate_single_take(take_path, debug=args.debug)
        
        if result:
            status = "PASSED" if result.get('overall_pass') else "FAILED"
            logger.info(f"Result: {status}")
            if result.get('report_html'):
                logger.info(f"Report: {result['report_html']}")
        
        return 0 if result and result.get('overall_pass') else 1
    
    # Batch mode
    batch = BatchValidator(data_dir, args.output)
    
    # List only mode
    if args.list_only:
        takes = batch.discovery.find_all_takes()
        print(f"\nFound {len(takes)} takes in {data_dir}:\n")
        for take in takes:
            rel_path = take.relative_to(data_dir)
            files = batch.discovery.get_take_files(take)
            if files:
                print(f"  {rel_path}/")
                print(f"    Video A: {files['video_a'].name}")
                print(f"    Video B: {files['video_b'].name}")
                print()
        return 0
    
    # Full batch processing
    logger.info(f"Starting batch validation of: {data_dir}")
    
    results = batch.validate_all_takes(debug=args.debug)
    
    if results:
        text_path, html_path = batch.generate_batch_summary()
        
        # Print summary
        passed = sum(1 for r in results if r.get('overall_pass', False))
        total = len(results)
        
        print("\n" + "=" * 60)
        print(f"BATCH COMPLETE: {passed}/{total} takes passed")
        print(f"Summary report: {html_path}")
        print("=" * 60)
        
        # Open HTML summary on macOS
        if sys.platform == 'darwin':
            os.system(f'open "{html_path}"')
        
        return 0 if passed == total else 1
    else:
        logger.warning("No takes found or processed")
        return 1


if __name__ == '__main__':
    exit(main())
