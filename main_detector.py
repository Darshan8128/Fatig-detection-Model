"""
════════════════════════════════════════════════════════════════
          STEP 10: REAL-TIME FATIGUE DETECTOR
════════════════════════════════════════════════════════════════
Live webcam detection with face mesh overlay, metrics, and alerts.
"""
import os
import sys
import cv2
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import tensorflow as tf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.face_analyzer import FaceAnalyzer
from core.fatigue_scorer import FatigueScorer, FatigueLevel
from system.screen_control import ScreenController
from system.session_logger import SessionLogger

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

# GPU config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RealtimeDetector:
    """Real-time fatigue detection with live visualization."""
    
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.fatigue_scorer = FatigueScorer()
        self.screen_controller = ScreenController()
        self.session_logger = SessionLogger()
        
        self.fatigue_model = None
        self.eye_model = None
        self.cap = None
        self.running = False
        self.paused = False
        
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        
        # ════════════════════════════════════════════
        # STABILITY IMPROVEMENTS - Prediction Buffers
        # ════════════════════════════════════════════
        self.cnn_fatigue_predictions = deque(maxlen=15)   # Smooth over 15 frames
        self.cnn_eye_predictions = deque(maxlen=20)       # Smooth over 20 frames
        self.fatigue_scores = deque(maxlen=10)            # Final score smoothing
        self.ear_values = deque(maxlen=30)                # EAR history for PERCLOS
        
        # Face tracking for stability
        self.last_face_box = None
        self.face_tracking_loss = 0
        self.max_tracking_loss = 5
        
        # Hysteresis for stable state transitions
        self.last_level = None
        self.level_stability_frames = 0
        self.min_level_frames = 3  # Require 3 consecutive frames before changing
        
        self.last_cnn_fatigue_prob = 0.5
        self.last_cnn_eye_prob = 0.5
        self.cnn_closed_eye_frames = 0
        self.fatigue_inference_interval = 2
        self.eye_inference_interval = 2
        self.capture_width = 640
        self.capture_height = 480
        
        self.session_start = time.time()
    
    def log(self, message):
        """Print message."""
        print(message)
    
    def setup_gpu(self):
        """Configure GPU."""
        self.log("\n GPU Configuration")
        
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        self.log(f"✅ {len(gpus)} GPU(s) configured")
    
    def load_models(self):
        """Load both CNN models."""
        self.log("\nLoading Models...")
        
        try:
            fatigue_path = MODELS_DIR / "fatigue_model.tflite"
            if not fatigue_path.exists():
                fatigue_path = MODELS_DIR / "fatigue_model.h5"
            
            if fatigue_path.suffix == '.h5':
                self.fatigue_model = tf.keras.models.load_model(str(fatigue_path))
            else:
                interpreter = tf.lite.Interpreter(str(fatigue_path))
                interpreter.allocate_tensors()
                self.fatigue_model = interpreter
            
            self.log(f"✅ Fatigue model loaded: {fatigue_path.name}")
        except Exception as e:
            self.log(f"❌ Failed to load fatigue model: {e}")
            return False
        
        try:
            eye_path = MODELS_DIR / "eye_model.tflite"
            if not eye_path.exists():
                eye_path = MODELS_DIR / "eye_model.h5"
            
            if eye_path.suffix == '.h5':
                self.eye_model = tf.keras.models.load_model(str(eye_path))
            else:
                interpreter = tf.lite.Interpreter(str(eye_path))
                interpreter.allocate_tensors()
                self.eye_model = interpreter
            
            self.log(f"✅ Eye model loaded: {eye_path.name}")
        except Exception as e:
            self.log(f"⚠️  Eye model not available: {e}")
            self.eye_model = None
        
        return True
    
    def predict_cnn(self, model, image, target_size):
        """Run CNN prediction on image."""
        try:
            if image is None or image.size == 0:
                return 0.5

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            img_batch = np.expand_dims(img, axis=0)
            
            if hasattr(model, 'predict'):
                output = model.predict(img_batch, verbose=0)[0]
                return float(output[0])
            else:
                # TFLite interpreter
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], img_batch)
                model.invoke()
                output = model.get_tensor(output_details[0]['index'])
                return float(output[0][0])
        except Exception as e:
            self.log(f"⚠️  CNN inference error: {e}")
            return 0.5

    def extract_eye_region(self, frame, face_metrics):
        """Extract a combined eye region for the eye-state model."""
        left_eye_box = face_metrics.get('left_eye_box')
        right_eye_box = face_metrics.get('right_eye_box')

        if left_eye_box and right_eye_box:
            lx, ly, lw, lh = left_eye_box
            rx, ry, rw, rh = right_eye_box
            x1 = max(0, min(lx, rx) - 10)
            y1 = max(0, min(ly, ry) - 10)
            x2 = min(frame.shape[1], max(lx + lw, rx + rw) + 10)
            y2 = min(frame.shape[0], max(ly + lh, ry + rh) + 10)
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]

        face_box = face_metrics.get('face_box')
        if face_box is None:
            return None

        x, y, w, h = face_box
        top = y + int(h * 0.12)
        bottom = y + int(h * 0.45)
        if bottom <= top:
            return None
        return frame[top:bottom, x:x+w]

    def extract_single_eye_region(self, frame, eye_box, padding=6):
        """Extract one eye crop with a small padding margin."""
        if eye_box is None:
            return None

        x, y, w, h = eye_box
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def predict_eye_openness(self, frame, face_metrics):
        """Predict eye openness from left/right eye crops and average them."""
        if self.eye_model is None:
            return self.last_cnn_eye_prob

        eye_predictions = []
        for key in ('left_eye_box', 'right_eye_box'):
            eye_roi = self.extract_single_eye_region(frame, face_metrics.get(key))
            if eye_roi is not None and eye_roi.size > 0:
                eye_predictions.append(self.predict_cnn(self.eye_model, eye_roi, (64, 64)))

        if eye_predictions:
            return float(np.mean(eye_predictions))

        eye_roi = self.extract_eye_region(frame, face_metrics)
        if eye_roi is not None and eye_roi.size > 0:
            return self.predict_cnn(self.eye_model, eye_roi, (64, 64))

        return self.last_cnn_eye_prob
    
    def get_smoothed_fatigue_prediction(self):
        """Get smoothed fatigue prediction from buffer (moving average)."""
        if not self.cnn_fatigue_predictions:
            return self.last_cnn_fatigue_prob
        return float(np.mean(self.cnn_fatigue_predictions))
    
    def get_smoothed_eye_prediction(self):
        """Get smoothed eye prediction from buffer (moving average)."""
        if not self.cnn_eye_predictions:
            return self.last_cnn_eye_prob
        return float(np.mean(self.cnn_eye_predictions))
    
    def calculate_perclos(self):
        """Calculate PERCLOS (% eye closure) from EAR history."""
        if not self.ear_values:
            return 0.0
        closed_count = sum(1 for e in self.ear_values if e < 0.15)
        return (closed_count / len(self.ear_values)) * 100
    
    def process_frame(self, frame):
        """Process single frame with ensemble smoothing."""
        # Face analysis
        face_metrics = self.face_analyzer.process_frame(frame)

        if not face_metrics['face_detected']:
            self.last_cnn_fatigue_prob = 0.5
            self.last_cnn_eye_prob = 0.5
            self.cnn_closed_eye_frames = 0
            self.last_face_box = None
            self.face_tracking_loss = 0
            return {
                'face_metrics': face_metrics,
                'fatigue_score': 0.0,
                'level': FatigueLevel.ALERT,
                'cnn_fatigue': None,
                'cnn_eye': None
            }
        
        # Face tracking for stability
        current_face_box = face_metrics['face_box']
        if self.last_face_box is not None:
            # Check if face moved too much (loss of tracking)
            x1, y1, w1, h1 = self.last_face_box
            x2, y2, w2, h2 = current_face_box
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (w2-w1)**2 + (h2-h1)**2)
            if distance > 200:  # Large movement = tracking loss
                self.face_tracking_loss += 1
            else:
                self.face_tracking_loss = 0
        
        self.last_face_box = current_face_box
        
        x, y, w, h = face_metrics['face_box']
        face_roi = frame[y:y+h, x:x+w]
        
        # ════════════════════════════════════════════
        # IMPROVED: CNN Predictions with Smoothing
        # ════════════════════════════════════════════
        
        should_run_fatigue_cnn = (
            self.fatigue_model is not None
            and (
                self.frame_count % self.fatigue_inference_interval == 0
                or not self.cnn_fatigue_predictions
            )
        )
        if should_run_fatigue_cnn:
            raw_fatigue_output = self.predict_cnn(
                self.fatigue_model, face_roi, (224, 224)
            )
            cnn_fatigue_prob = 1.0 - raw_fatigue_output  # Convert to fatigue prob
            self.cnn_fatigue_predictions.append(cnn_fatigue_prob)
        
        should_run_eye_cnn = (
            self.eye_model is not None
            and (
                self.frame_count % self.eye_inference_interval == 0
                or not self.cnn_eye_predictions
            )
        )
        if should_run_eye_cnn:
            raw_eye_output = self.predict_eye_openness(frame, face_metrics)
            cnn_eye_open_prob = 1.0 - raw_eye_output
            self.cnn_eye_predictions.append(cnn_eye_open_prob)
        
        # Get smoothed predictions (moving average)
        cnn_fatigue_smoothed = self.get_smoothed_fatigue_prediction()
        cnn_eye_smoothed = self.get_smoothed_eye_prediction()
        self.last_cnn_fatigue_prob = cnn_fatigue_smoothed
        self.last_cnn_eye_prob = cnn_eye_smoothed
        
        # Update eye closure tracking with smoothed prediction
        if cnn_eye_smoothed < 0.40:  # Eyes clearly open
            self.cnn_closed_eye_frames = 0
        elif cnn_eye_smoothed > 0.60:  # Eyes clearly closed
            self.cnn_closed_eye_frames += 1
        else:  # Uncertain - use hysteresis
            if self.cnn_closed_eye_frames > 0:
                self.cnn_closed_eye_frames += 1
            else:
                self.cnn_closed_eye_frames = max(0, self.cnn_closed_eye_frames - 1)

        face_metrics['cnn_closed_eye_frames'] = self.cnn_closed_eye_frames
        
        # Track EAR for PERCLOS calculation
        ear = face_metrics.get('ear', 0.0)
        self.ear_values.append(ear)
        
        # ════════════════════════════════════════════
        # IMPROVED: Ensemble Scoring
        # ════════════════════════════════════════════
        # Use smoothed predictions + manual metrics
        fatigue_score = self.fatigue_scorer.calculate_fatigue_score(
            face_metrics, cnn_fatigue_smoothed, cnn_eye_smoothed
        )
        
        # Apply final smoothing to fatigue score
        self.fatigue_scores.append(fatigue_score)
        smoothed_fatigue_score = float(np.mean(self.fatigue_scores))
        
        # ════════════════════════════════════════════
        # IMPROVED: State Stability
        # ════════════════════════════════════════════
        level = self.fatigue_scorer.update_state(smoothed_fatigue_score)
        
        # Apply hysteresis to prevent rapid state changes
        if level != self.last_level:
            self.level_stability_frames = 1
        else:
            self.level_stability_frames += 1
        
        # Only change state after seeing it for min_level_frames consecutive frames
        if self.level_stability_frames < self.min_level_frames:
            level = self.last_level if self.last_level is not None else level
        else:
            self.last_level = level
        
        return {
            'face_metrics': face_metrics,
            'fatigue_score': smoothed_fatigue_score,
            'level': level,
            'cnn_fatigue': cnn_fatigue_smoothed,
            'cnn_eye': cnn_eye_smoothed
        }
    
    def draw_metrics_overlay(self, frame, result):
        """Draw metrics overlay on frame."""
        h, w = frame.shape[:2]
        
        if not result['face_metrics']['face_detected']:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Get level color
        level_color = self.fatigue_scorer.get_level_color(result['level'])
        level_name = self.fatigue_scorer.get_level_name(result['level'])
        score = result['fatigue_score']
        confidence = score * 100
        
        face_box = result['face_metrics'].get('face_box')
        if face_box is not None:
            x, y, fw, fh = face_box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), level_color, 3)
            cv2.putText(frame, "FACE", (x, max(25, y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, level_color, 2)
        
        left_eye_box = result['face_metrics'].get('left_eye_box')
        if left_eye_box is not None:
            ex, ey, ew, eh = left_eye_box
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            cv2.putText(frame, "L-EYE", (ex, max(20, ey - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        right_eye_box = result['face_metrics'].get('right_eye_box')
        if right_eye_box is not None:
            ex, ey, ew, eh = right_eye_box
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            cv2.putText(frame, "R-EYE", (ex, max(20, ey - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        cv2.rectangle(frame, (0, 0), (w, 145), (30, 30, 30), -1)
        
        # HUD - Top left
        y = 40
        cv2.putText(frame, f"Level: {level_name}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, level_color, 2)
        y += 35
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 35
        
        # Fatigue score bar
        bar_width = 300
        bar_height = 20
        bar_x = 15
        filled_width = int(bar_width * min(score, 1.0))
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                     (200, 200, 200), 2)
        if filled_width > 0:
            cv2.rectangle(frame, (bar_x, y), (bar_x + filled_width, y + bar_height),
                         level_color, -1)
        cv2.putText(frame, f"{score*100:.1f}%", (bar_x + bar_width + 10, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 45
        
        # Metrics
        ear = result['face_metrics']['ear']
        mar = result['face_metrics']['mar']
        perclos = result['face_metrics']['perclos']
        blink_rate = result['face_metrics']['blink_rate']
        yawns = result['face_metrics']['yawn_count']
        cnn_closed_eye_frames = result['face_metrics'].get('cnn_closed_eye_frames', 0)
        measured_closed_eye_frames = result['face_metrics'].get('closed_eye_frames', 0)
        
        cv2.putText(frame, f"EAR: {ear:.3f}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}", (15, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (200, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Blinks/min: {blink_rate:.0f}", (200, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Yawns: {yawns}", (400, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Face CNN: {result['cnn_fatigue'] * 100:.1f}%" if result['cnn_fatigue'] is not None else "Face CNN: N/A", (560, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Eye CNN: {result['cnn_eye'] * 100:.1f}%" if result['cnn_eye'] is not None else "Eye CNN: N/A", (560, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Closed Eye Frames: {measured_closed_eye_frames}", (820, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Eye CNN Closed: {cnn_closed_eye_frames}", (820, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Session timer
        elapsed = int(time.time() - self.session_start)
        minutes = elapsed // 60
        seconds = elapsed % 60
        cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", (400, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # FPS - Top right
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timer > 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = now

    def open_webcam(self, camera_indexes=None):
        """Open the first available webcam from a list of indexes."""
        indexes = camera_indexes or [0, 1, 2, 3]
        backends = [
            ("default", cv2.CAP_ANY),
            ("msmf", cv2.CAP_MSMF),
            ("dshow", cv2.CAP_DSHOW),
        ]

        for backend_name, backend in backends:
            for camera_index in indexes:
                self.log(f"  Trying camera index {camera_index} with {backend_name}...")
                cap = cv2.VideoCapture(camera_index, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        self.log(
                            f"✅ Webcam opened on camera index {camera_index} using {backend_name}"
                        )
                        return cap

                cap.release()

        return None
    
    def run(self):
        """Main detection loop."""
        self.log("\n" + "=" * 70)
        self.log("REAL-TIME FATIGUE DETECTION")
        self.log("=" * 70)
        
        # Setup
        self.setup_gpu()
        if not self.load_models():
            return False
        
        self.log("\nStarting webcam...")
        self.cap = self.open_webcam()
        
        if self.cap is None or not self.cap.isOpened():
            self.log("❌ Failed to open webcam on indexes 0-3 with default/msmf/dshow!")
            return False

        self.log("\nDetection started. Controls:")
        self.log("  Q - Quit & save session")
        self.log("  P - Pause/Resume")
        self.log("  R - Reset stats")
        self.log("  S - Screenshot\n")
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if not self.paused:
                    # Process frame
                    result = self.process_frame(frame)
                    
                    # Log to session
                    self.session_logger.log_frame(
                        self.frame_count,
                        result['fatigue_score'],
                        result['level'],
                        result['face_metrics'],
                        result['cnn_fatigue'],
                        result['cnn_eye']
                    )
                    
                    # Update screen controller
                    self.screen_controller.update(result['level'])
                    
                    self.frame_count += 1
                
                # Draw overlay
                frame = self.draw_metrics_overlay(frame, result if not self.paused else {
                    'face_metrics': {'face_detected': False},
                    'fatigue_score': 0,
                    'level': FatigueLevel.ALERT,
                    'cnn_fatigue': None,
                    'cnn_eye': None
                })
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('Fatigue Detection', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('p'):
                    self.paused = not self.paused
                    self.log(f"{'⏸️  Paused' if self.paused else '▶️  Resumed'}")
                elif key == ord('r'):
                    self.face_analyzer = FaceAnalyzer()
                    self.fatigue_scorer = FatigueScorer()
                    self.log("🔄 Stats reset")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = PROJECT_ROOT / f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                    self.log(f"📸 Screenshot: {screenshot_path}")
        
        except KeyboardInterrupt:
            self.log("\n⏹️  Interrupted by user")
        
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()
            
            # End session
            self.log("\n" + "=" * 70)
            self.log("SESSION ENDED")
            self.log("=" * 70)
            summary = self.session_logger.end_session()
            
            self.log(f"\n✅ Session saved to logs/")
            self.log(f"   • Duration: {summary['session_info']['duration_minutes']} minutes")
            self.log(f"   • Frames: {summary['session_info']['total_frames']}")
            self.log(f"   • Peak Score: {summary['fatigue_statistics']['peak_fatigue_score']}\n")
        
        return True


def main():
    """Main entry point."""
    try:
        detector = RealtimeDetector()
        detector.run()
        print("\n✅ [main_detector.py] complete. Type NEXT for next file.\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
