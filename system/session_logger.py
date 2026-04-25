"""
════════════════════════════════════════════════════════════════
     STEP 9: SESSION LOGGER
════════════════════════════════════════════════════════════════
Log fatigue metrics to CSV and JSON session summaries.
"""
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from core.fatigue_scorer import FatigueLevel


class SessionLogger:
    """Log fatigue detection sessions."""
    
    def __init__(self, logs_dir=None):
        """Initialize logger."""
        if logs_dir is None:
            logs_dir = Path(__file__).parent.parent / "logs"
        
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Start timestamp
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV file
        self.csv_file = self.logs_dir / f"session_{timestamp}.csv"
        self.csv_writer = None
        self.csv_handle = None
        
        # Session data
        self.session_data = {
            'timestamp': timestamp,
            'start_datetime': datetime.now().isoformat(),
            'frames_logged': 0,
            'fatigue_scores': [],
            'levels': {},
            'metrics': {
                'ear_values': [],
                'mar_values': [],
                'perclos_values': [],
                'blink_rates': [],
                'yawn_counts': []
            },
            'events': [],
            'level_durations': {
                'ALERT': 0,
                'MILD': 0,
                'MODERATE': 0,
                'SEVERE': 0
            }
        }
        
        self.last_level = None
        self.level_start_time = time.time()
        
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file."""
        try:
            self.csv_handle = open(self.csv_file, 'w', newline='')
            self.csv_writer = csv.DictWriter(
                self.csv_handle,
                fieldnames=[
                    'timestamp', 'frame_num', 'fatigue_score', 'level',
                    'ear', 'mar', 'perclos', 'blink_rate', 'yawn_count',
                    'head_pitch', 'cnn_fatigue', 'cnn_eye'
                ]
            )
            self.csv_writer.writeheader()
            self.csv_handle.flush()
            print(f"✅ Session CSV started: {self.csv_file}")
        except Exception as e:
            print(f"❌ Failed to init CSV: {e}")
    
    def log_frame(self, frame_num, fatigue_score, level, face_metrics,
                  cnn_fatigue_prob=None, cnn_eye_prob=None):
        """
        Log a single frame.
        
        Args:
            frame_num: Frame number
            fatigue_score: Calculated fatigue score [0-1]
            level: FatigueLevel enum
            face_metrics: Dict from FaceAnalyzer
            cnn_fatigue_prob: CNN fatigue output [0-1]
            cnn_eye_prob: CNN eye output [0-1]
        """
        try:
            # Get current time relative to start
            elapsed = time.time() - self.start_time
            
            # Extract metrics
            ear = face_metrics.get('ear', 0)
            mar = face_metrics.get('mar', 0)
            perclos = face_metrics.get('perclos', 0)
            blink_rate = face_metrics.get('blink_rate', 0)
            yawn_count = face_metrics.get('yawn_count', 0)
            head_pitch = face_metrics.get('head_pitch', 0)
            
            # Get level name
            level_name = self._get_level_name(level)
            
            # Write to CSV
            if self.csv_writer:
                self.csv_writer.writerow({
                    'timestamp': f"{elapsed:.2f}s",
                    'frame_num': frame_num,
                    'fatigue_score': f"{fatigue_score:.4f}",
                    'level': level_name,
                    'ear': f"{ear:.4f}",
                    'mar': f"{mar:.4f}",
                    'perclos': f"{perclos:.2f}",
                    'blink_rate': f"{blink_rate:.1f}",
                    'yawn_count': yawn_count,
                    'head_pitch': f"{head_pitch:.1f}",
                    'cnn_fatigue': f"{cnn_fatigue_prob:.4f}" if cnn_fatigue_prob is not None else "N/A",
                    'cnn_eye': f"{cnn_eye_prob:.4f}" if cnn_eye_prob is not None else "N/A"
                })
            
            # Update session statistics
            self.session_data['fatigue_scores'].append(fatigue_score)
            self.session_data['metrics']['ear_values'].append(ear)
            self.session_data['metrics']['mar_values'].append(mar)
            self.session_data['metrics']['perclos_values'].append(perclos)
            self.session_data['metrics']['blink_rates'].append(blink_rate)
            self.session_data['metrics']['yawn_counts'].append(yawn_count)
            self.session_data['frames_logged'] += 1
            
            # Track level changes
            if level != self.last_level:
                # Record duration of previous level
                if self.last_level is not None:
                    level_name = self._get_level_name(self.last_level)
                    duration = time.time() - self.level_start_time
                    self.session_data['level_durations'][level_name] += duration
                    
                    # Log level change event
                    new_level_name = self._get_level_name(level)
                    self.session_data['events'].append({
                        'type': 'level_change',
                        'time': f"{elapsed:.2f}s",
                        'frame': frame_num,
                        'from': level_name,
                        'to': new_level_name,
                        'fatigue_score': fatigue_score
                    })
                
                self.last_level = level
                self.level_start_time = time.time()
            
            # Flush CSV every 30 frames
            if self.session_data['frames_logged'] % 30 == 0:
                if self.csv_handle:
                    self.csv_handle.flush()
        
        except Exception as e:
            print(f"❌ Error logging frame: {e}")
    
    def log_event(self, event_type, description, frame_num=None, data=None):
        """Log a custom event."""
        elapsed = time.time() - self.start_time
        event = {
            'type': event_type,
            'time': f"{elapsed:.2f}s",
            'frame': frame_num,
            'description': description,
            'data': data
        }
        self.session_data['events'].append(event)
    
    def end_session(self):
        """Finalize session and save JSON summary."""
        elapsed_seconds = time.time() - self.start_time

        # Close CSV
        if self.csv_handle:
            self.csv_handle.close()
            self.csv_handle = None
        
        # Calculate final level duration
        if self.last_level is not None:
            level_name = self._get_level_name(self.last_level)
            duration = time.time() - self.level_start_time
            self.session_data['level_durations'][level_name] += duration
        
        # Calculate statistics
        total_frames = self.session_data['frames_logged']
        duration_minutes = elapsed_seconds / 60 if elapsed_seconds > 0 else 0

        scores = self.session_data['fatigue_scores']
        peak_score = max(scores) if scores else 0
        peak_frame = scores.index(peak_score) if scores else 0

        time_per_level = {}
        for level_name, duration in self.session_data['level_durations'].items():
            percentage = (duration / elapsed_seconds * 100) if elapsed_seconds > 0 else 0
            time_per_level[level_name] = {
                'seconds': f"{duration:.1f}",
                'percentage': f"{percentage:.1f}%"
            }
        
        ears = self.session_data['metrics']['ear_values']
        blinks = self.session_data['metrics']['blink_rates']
        yawns = self.session_data['metrics']['yawn_counts']
        
        avg_ear = sum(ears) / len(ears) if ears else 0
        avg_blink = sum(blinks) / len(blinks) if blinks else 0
        total_yawns = yawns[-1] if yawns else 0
        
        # Build summary
        summary = {
            'session_info': {
                'timestamp': self.session_data['timestamp'],
                'start_datetime': self.session_data['start_datetime'],
                'end_datetime': datetime.now().isoformat(),
                'duration_minutes': f"{duration_minutes:.2f}",
                'total_frames': total_frames,
                'csv_file': str(self.csv_file)
            },
            'fatigue_statistics': {
                'peak_fatigue_score': f"{peak_score:.4f}",
                'peak_frame': peak_frame,
                'average_fatigue_score': f"{sum(scores) / len(scores):.4f}" if scores else 0,
                'min_fatigue_score': f"{min(scores):.4f}" if scores else 0,
                'max_fatigue_score': f"{max(scores):.4f}" if scores else 0
            },
            'metrics': {
                'average_ear': f"{avg_ear:.4f}",
                'average_blink_rate': f"{avg_blink:.1f}",
                'total_yawns': int(total_yawns)
            },
            'time_per_level': time_per_level,
            'notable_events': self.session_data['events'][:20]  # First 20 events
        }
        
        # Save JSON
        timestamp = self.session_data['timestamp']
        json_file = self.logs_dir / f"summary_{timestamp}.json"
        
        try:
            with open(json_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✅ Session summary saved: {json_file}")
        except Exception as e:
            print(f"❌ Failed to save session summary: {e}")
        
        return summary
    
    def _get_level_name(self, level):
        """Get level name from enum."""
        level_names = {
            FatigueLevel.ALERT: "ALERT",
            FatigueLevel.MILD: "MILD",
            FatigueLevel.MODERATE: "MODERATE",
            FatigueLevel.SEVERE: "SEVERE"
        }
        return level_names.get(level, "UNKNOWN")


if __name__ == "__main__":
    print("✅ system/session_logger.py complete. Type NEXT for next file.\n")
