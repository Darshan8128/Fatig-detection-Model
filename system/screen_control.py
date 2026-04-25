"""
════════════════════════════════════════════════════════════════
     STEP 8: SCREEN CONTROLLER - WINDOWS 11 ALERTS
════════════════════════════════════════════════════════════════
Control brightness, sounds, and notifications based on fatigue level.
"""
import time
import winsound
import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path
from core.fatigue_scorer import FatigueLevel

try:
    from screen_brightness_control import set_brightness, get_brightness
except ImportError:
    set_brightness = None
    get_brightness = None

try:
    from win10toast import ToastNotifier
except ImportError:
    ToastNotifier = None


class ScreenController:
    """Control Windows 11 screen for fatigue alerts."""
    
    def __init__(self):
        """Initialize screen controller."""
        self.current_level = FatigueLevel.ALERT
        self.current_brightness = 100
        self.target_brightness = 100
        self.transition_steps = 0
        self.transition_max_steps = 30  # Smooth over 1 second (30fps)
        self.severe_popup_time = None
        self.severe_popup = None
        self.snooze_until = None
    
    def log(self, message):
        """Print log message."""
        print(f"[ScreenController] {message}")
    
    def set_brightness(self, brightness):
        """Set screen brightness (0-100)."""
        if set_brightness is None:
            return
        
        try:
            set_brightness(brightness)
            self.current_brightness = brightness
        except Exception as e:
            self.log(f"⚠️  Failed to set brightness: {e}")
    
    def smooth_brightness_transition(self, target):
        """Gradually transition brightness to target."""
        if self.current_brightness == target:
            return
        
        # Calculate step size
        diff = target - self.current_brightness
        if diff > 0:
            step = max(1, diff // self.transition_max_steps)
        else:
            step = min(-1, diff // self.transition_max_steps)
        
        new_brightness = self.current_brightness + step
        if (step > 0 and new_brightness >= target) or (step < 0 and new_brightness <= target):
            new_brightness = target
        
        self.set_brightness(new_brightness)
    
    def play_beep(self, frequency=800, duration=200):
        """Play beep sound."""
        try:
            winsound.Beep(frequency, duration)
        except Exception as e:
            self.log(f"⚠️  Failed to play beep: {e}")
    
    def play_alarm(self):
        """Play loud alarm sound."""
        try:
            # 1000Hz x 3 beeps
            for _ in range(3):
                winsound.Beep(1000, 500)
                time.sleep(0.2)
        except Exception as e:
            self.log(f"⚠️  Failed to play alarm: {e}")
    
    def show_notification(self, title, message):
        """Show Windows desktop notification."""
        if ToastNotifier is None:
            self.log(f"Notification: {title} - {message}")
            return
        
        try:
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=5)
        except Exception as e:
            self.log(f"⚠️  Failed to show notification: {e}")
    
    def show_severe_popup(self):
        """Show fullscreen SEVERE FATIGUE popup."""
        try:
            self.severe_popup = tk.Tk()
            self.severe_popup.attributes('-fullscreen', True)
            self.severe_popup.configure(bg='red')
            
            # Main label
            main_font = tkfont.Font(family="Arial", size=72, weight="bold")
            label = tk.Label(
                self.severe_popup,
                text="⚠️ SEVERE FATIGUE DETECTED",
                font=main_font,
                fg='white',
                bg='red'
            )
            label.pack(expand=True)
            
            # Instruction label
            instr_font = tkfont.Font(family="Arial", size=36)
            instr_label = tk.Label(
                self.severe_popup,
                text="Please take a break immediately!",
                font=instr_font,
                fg='white',
                bg='red'
            )
            instr_label.pack(expand=True)
            
            # Buttons frame
            button_frame = tk.Frame(self.severe_popup, bg='red')
            button_frame.pack(expand=True, pady=50)
            
            button_font = tkfont.Font(family="Arial", size=24)
            
            take_break_btn = tk.Button(
                button_frame,
                text="Take Break",
                font=button_font,
                bg='green',
                fg='white',
                padx=50,
                pady=30,
                command=self.on_take_break
            )
            take_break_btn.pack(side=tk.LEFT, padx=30)
            
            snooze_btn = tk.Button(
                button_frame,
                text="Snooze 5min",
                font=button_font,
                bg='orange',
                fg='white',
                padx=50,
                pady=30,
                command=self.on_snooze
            )
            snooze_btn.pack(side=tk.RIGHT, padx=30)
            
            self.severe_popup_time = time.time()
            self.severe_popup.after(100, self.severe_popup.lift)
            self.severe_popup.after(100, self.severe_popup.focus_set)
            
            self.log("Severe fatigue popup displayed")
        
        except Exception as e:
            self.log(f"⚠️  Failed to show popup: {e}")
    
    def on_take_break(self):
        """User clicked Take Break."""
        self.log("User taking a break")
        self.close_severe_popup()
    
    def on_snooze(self):
        """User clicked Snooze."""
        self.snooze_until = time.time() + 300  # 5 minutes
        self.log("Snoozed for 5 minutes")
        self.close_severe_popup()
    
    def close_severe_popup(self):
        """Close severe popup."""
        if self.severe_popup:
            try:
                self.severe_popup.destroy()
                self.severe_popup = None
                self.severe_popup_time = None
            except:
                pass
    
    def update_for_level(self, level):
        """Update screen for given fatigue level."""
        self.current_level = level
        
        if level == FatigueLevel.ALERT:
            self.smooth_brightness_transition(100)
        
        elif level == FatigueLevel.MILD:
            self.smooth_brightness_transition(75)
            if self.current_brightness == 75:
                # First time reaching MILD
                if not hasattr(self, '_mild_notified') or not self._mild_notified:
                    self.show_notification(
                        "Mild Fatigue Detected",
                        "Please stay alert. Consider taking a break soon."
                    )
                    self._mild_notified = True
        
        elif level == FatigueLevel.MODERATE:
            self.smooth_brightness_transition(50)
            self.play_beep(800, 200)
            if self.current_brightness == 50:
                if not hasattr(self, '_moderate_notified') or not self._moderate_notified:
                    self.show_notification(
                        "Moderate Fatigue Detected",
                        "Take a break now. Your eyes need rest."
                    )
                    self._moderate_notified = True
        
        elif level == FatigueLevel.SEVERE:
            # Check snooze
            if self.snooze_until and time.time() < self.snooze_until:
                self.smooth_brightness_transition(25)
                return
            
            self.smooth_brightness_transition(25)
            self.play_alarm()
            
            # Show popup if not already shown
            if not self.severe_popup:
                self.show_severe_popup()
            
            # Check if popup ignored for 30 seconds
            if self.severe_popup_time:
                if time.time() - self.severe_popup_time > 30:
                    self.smooth_brightness_transition(10)
    
    def reset_notifications(self):
        """Reset notification flags."""
        self._mild_notified = False
        self._moderate_notified = False
    
    def update(self, level):
        """Update controller state (call every frame)."""
        self.update_for_level(level)


if __name__ == "__main__":
    print("✅ system/screen_control.py complete. Type NEXT for next file.\n")
