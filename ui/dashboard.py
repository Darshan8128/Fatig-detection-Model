"""
════════════════════════════════════════════════════════════════
          STEP 11: TKINTER LIVE DASHBOARD
════════════════════════════════════════════════════════════════
Live dashboard with gauge, metrics, and graphs.
"""
import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path
from collections import deque
import time
import threading

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf

from core.face_analyzer import FaceAnalyzer
from core.fatigue_scorer import FatigueScorer, FatigueLevel

PROJECT_ROOT = Path(__file__).parent


class Dashboard:
    """Tkinter-based live dashboard."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Fatigue Detection Dashboard")
        self.root.geometry("1400x900")
        
        self.face_analyzer = FaceAnalyzer()
        self.fatigue_scorer = FatigueScorer()
        
        self.running = False
        self.paused = False
        self.cap = None
        self.frame_count = 0
        
        # Data history
        self.score_history = deque(maxlen=300)  # 5 min at 30fps
        self.level_time = {'ALERT': 0, 'MILD': 0, 'MODERATE': 0, 'SEVERE': 0}
        self.ear_history = deque(maxlen=100)
        self.blink_history = deque(maxlen=100)
        
        # UI setup
        self.setup_ui()
    
    def setup_ui(self):
        """Setup dashboard UI."""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Metrics
        left_frame = tk.Frame(main_frame, bg='#252525', relief=tk.SUNKEN, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Title
        title_font = tkfont.Font(family="Arial", size=16, weight="bold")
        title = tk.Label(left_frame, text="Live Metrics", font=title_font,
                        fg='white', bg='#252525')
        title.pack(pady=10)
        
        # Fatigue gauge (simple text + color-coded bar)
        gauge_frame = tk.Frame(left_frame, bg='#252525')
        gauge_frame.pack(fill=tk.X, padx=20, pady=20)
        
        gauge_label = tk.Label(gauge_frame, text="Fatigue Level", font=("Arial", 12),
                              fg='white', bg='#252525')
        gauge_label.pack(anchor=tk.W)
        
        self.level_label = tk.Label(gauge_frame, text="ALERT", font=("Arial", 36, "bold"),
                                   fg='#00FF00', bg='#252525')
        self.level_label.pack(anchor=tk.W, pady=5)
        
        self.score_var = tk.StringVar(value="Score: 0.00%")
        self.score_label = tk.Label(gauge_frame, textvariable=self.score_var,
                                   font=("Arial", 24), fg='white', bg='#252525')
        self.score_label.pack(anchor=tk.W)
        
        # Metrics section
        metrics_frame = tk.LabelFrame(left_frame, text="Facial Metrics", font=("Arial", 12),
                                     fg='white', bg='#252525', relief=tk.FLAT)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.metrics_vars = {}
        metrics_list = [
            ('EAR', 'ear_val'),
            ('MAR', 'mar_val'),
            ('PERCLOS', 'perclos_val'),
            ('Blinks/min', 'blink_val'),
            ('Yawns', 'yawn_val'),
            ('Head Pitch', 'pitch_val')
        ]
        
        for label, key in metrics_list:
            self.metrics_vars[key] = tk.StringVar(value="--")
            frame = tk.Frame(metrics_frame, bg='#252525')
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=f"{label}:", font=("Arial", 11),
                    fg='#CCCCCC', bg='#252525', width=12, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(frame, textvariable=self.metrics_vars[key], font=("Arial", 11),
                    fg='#00FF00', bg='#252525').pack(side=tk.LEFT)
        
        # Timer section
        timer_frame = tk.LabelFrame(left_frame, text="Session", font=("Arial", 12),
                                   fg='white', bg='#252525', relief=tk.FLAT)
        timer_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.timer_var = tk.StringVar(value="00:00:00")
        timer_label = tk.Label(timer_frame, textvariable=self.timer_var,
                              font=("Arial", 28, "bold"), fg='#00FF00', bg='#252525')
        timer_label.pack(pady=10)
        
        # Right panel - Graphs
        right_frame = tk.Frame(main_frame, bg='#252525', relief=tk.SUNKEN, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Graphs
        self.fig = Figure(figsize=(7, 8), dpi=100, facecolor='#3e3e3e', edgecolor='#555555')
        self.fig.patch.set_facecolor('#252525')
        
        # Fatigue score line graph
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax1.set_title("Fatigue Score (Last 5 min)", color='white', fontsize=10)
        self.ax1.set_ylim([0, 1])
        self.ax1.set_facecolor('#2d2d2d')
        self.ax1.tick_params(colors='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        self.line1, = self.ax1.plot([], [], color='#FF9500', linewidth=2)
        
        # Level distribution
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax2.set_title("Time per Level", color='white', fontsize=10)
        self.ax2.set_facecolor('#2d2d2d')
        self.ax2.tick_params(colors='white')
        self.bars = None
        
        # EAR history
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        self.ax3.set_title("Eye Aspect Ratio (EAR)", color='white', fontsize=10)
        self.ax3.set_ylim([0, 0.5])
        self.ax3.set_facecolor('#2d2d2d')
        self.ax3.tick_params(colors='white')
        self.ax3.spines['bottom'].set_color('white')
        self.ax3.spines['left'].set_color('white')
        self.line3, = self.ax3.plot([], [], color='#00FF00', linewidth=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#1e1e1e')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        start_btn = tk.Button(button_frame, text="Start Detection", command=self.start_detection,
                             bg='#00AA00', fg='white', font=("Arial", 11, "bold"), padx=15, pady=5)
        start_btn.pack(side=tk.LEFT, padx=5)
        
        stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_detection,
                            bg='#AA0000', fg='white', font=("Arial", 11, "bold"), padx=15, pady=5)
        stop_btn.pack(side=tk.LEFT, padx=5)
        
        reset_btn = tk.Button(button_frame, text="Reset", command=self.reset_stats,
                             bg='#666666', fg='white', font=("Arial", 11, "bold"), padx=15, pady=5)
        reset_btn.pack(side=tk.LEFT, padx=5)
    
    def start_detection(self):
        """Start detection thread."""
        if self.running:
            return
        
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.session_start = time.time()
        
        # Start detection thread
        thread = threading.Thread(target=self.detection_loop, daemon=True)
        thread.start()
    
    def detection_loop(self):
        """Run detection in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            metrics = self.face_analyzer.process_frame(frame)
            
            if metrics['face_detected']:
                fatigue_score = self.fatigue_scorer.calculate_fatigue_score(metrics, 0.5, 0.5)
                level = self.fatigue_scorer.update_state(fatigue_score)
                
                self.score_history.append(fatigue_score)
                self.ear_history.append(metrics['ear'])
                self.blink_history.append(metrics['blink_rate'])
                
                # Update level timing
                level_name = self.fatigue_scorer.get_level_name(level)
                self.level_time[level_name] += 1
                
                # Update UI
                self.update_metrics(metrics, fatigue_score, level)
            
            self.frame_count += 1
            time.sleep(0.03)  # ~30fps
    
    def update_metrics(self, metrics, score, level):
        """Update UI with metrics."""
        self.metrics_vars['ear_val'].set(f"{metrics['ear']:.3f}")
        self.metrics_vars['mar_val'].set(f"{metrics['mar']:.3f}")
        self.metrics_vars['perclos_val'].set(f"{metrics['perclos']:.1f}%")
        self.metrics_vars['blink_val'].set(f"{metrics['blink_rate']:.0f}")
        self.metrics_vars['yawn_val'].set(f"{metrics['yawn_count']}")
        self.metrics_vars['pitch_val'].set(f"{metrics['head_pitch']:.1f}°")
        
        level_name = self.fatigue_scorer.get_level_name(level)
        self.level_label.config(text=level_name)
        
        color = {
            'ALERT': '#00FF00',
            'MILD': '#FFFF00',
            'MODERATE': '#FF9500',
            'SEVERE': '#FF0000'
        }.get(level_name, '#FFFFFF')
        
        self.level_label.config(fg=color)
        self.score_var.set(f"Score: {score*100:.1f}%")
        
        # Update timer
        elapsed = int(time.time() - self.session_start)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self.timer_var.set(f"{h:02d}:{m:02d}:{s:02d}")
        
        # Update graphs
        self.update_graphs()
    
    def update_graphs(self):
        """Update graph visualizations."""
        # Score line
        if len(self.score_history) > 0:
            self.line1.set_data(range(len(self.score_history)), list(self.score_history))
            self.ax1.set_xlim([0, max(300, len(self.score_history))])
        
        # Level distribution
        total_frames = sum(self.level_time.values()) or 1
        levels = list(self.level_time.keys())
        values = [self.level_time[l] / total_frames for l in levels]
        colors = ['#00FF00', '#FFFF00', '#FF9500', '#FF0000']
        
        if self.bars:
            for bar in self.bars:
                bar.remove()
        self.bars = self.ax2.bar(levels, values, color=colors)
        
        # EAR history
        if len(self.ear_history) > 0:
            self.line3.set_data(range(len(self.ear_history)), list(self.ear_history))
            self.ax3.set_xlim([0, max(100, len(self.ear_history))])
        
        self.canvas.draw_idle()
    
    def stop_detection(self):
        """Stop detection."""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def reset_stats(self):
        """Reset statistics."""
        self.score_history.clear()
        self.ear_history.clear()
        self.blink_history.clear()
        self.level_time = {'ALERT': 0, 'MILD': 0, 'MODERATE': 0, 'SEVERE': 0}
        self.frame_count = 0
        self.session_start = time.time()


def main():
    """Main entry point."""
    root = tk.Tk()
    root.configure(bg='#1e1e1e')
    dashboard = Dashboard(root)
    root.mainloop()
    print("\n✅ [ui/dashboard.py] complete. Type NEXT for next file.\n")


if __name__ == "__main__":
    main()
