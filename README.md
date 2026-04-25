his file records the full build flow of the project, including the original implementation steps and the later fixes made during live testing.

1. Project Setup

1. Created the project workspace at c:\Users\darsh\OneDrive\Desktop\Fatigue2.
2. Created and activated a Python environment.
3. Installed dependencies from requirements.txt.
4. Verified the project folders:
   - Dataset/
   - config/
   - core/
   - models/
   - system/
   - ui/
5. Added the main scripts for dataset prep, training, evaluation, calibration, and live detection.

2. Dataset Exploration

1. Scanned all dataset archives with explore_dataset.py.
2. Counted images in the fatigue, non-fatigue, eye-open, and eye-closed classes.
3. Checked for missing or corrupt files.
4. Recorded the dataset summary in dataset_report.txt.

3. Dataset Preparation

1. Mapped raw folders into project labels:
   - fatigue
   - non_fatigue
   - eye_open
   - eye_closed
2. Merged source images into processed_dataset/.
3. Resized images for the two model pipelines:
   - fatigue model input: 224x224
   - eye model input: 64x64
4. Balanced the class distributions.
5. Split the dataset into:
   - train
   - val
   - test
6. Saved mapping and split details in:
   - config/dataset_mapping.json
   - config/split_summary.json

4. Fatigue Model Training

1. Built train_fatigue_model.py.
2. Used MobileNetV2 transfer learning for fatigue vs non-fatigue.
3. Loaded data from final_dataset/train/fatigue and final_dataset/train/non_fatigue.
4. Applied preprocessing and augmentation.
5. Trained the classifier and saved:
   - models/fatigue_model.h5
   - models/fatigue_model.tflite
   - models/fatigue_history.json

5. Eye Model Training

1. Built train_eye_model.py.
2. Used MobileNetV2 alpha 0.5 for eye-open vs eye-closed.
3. Loaded data from final_dataset/train/eye_open and final_dataset/train/eye_closed.
4. Trained the model and saved:
   - models/eye_model.h5
   - models/eye_model.tflite
   - models/eye_history.json
5. Training labels were set as:
   - eye_open = 1
   - eye_closed = 0

6. Model Evaluation

1. Built evaluate_models.py.
2. Evaluated both saved models on the validation and test splits.
3. Generated reports and plots in reports/.
4. Confirmed the models were usable for the live pipeline.

7. Core Runtime Modules

Face Analyzer

1. Built core/face_analyzer.py.
2. Used OpenCV Haar cascades for:
   - face detection
   - eye detection
3. Computed runtime metrics:
   - EAR-like eye openness estimate
   - PERCLOS
   - blink rate
   - basic head yaw
4. Added face box output.
5. Added left-eye box output.
6. Added right-eye box output.

Fatigue Scorer

1. Built core/fatigue_scorer.py.
2. Combined:
   - fatigue CNN score
   - eye CNN score
   - EAR score
   - PERCLOS
   - blink rate
   - yawn score
   - head pose score
3. Added score smoothing.
4. Added fatigue levels:
   - ALERT
   - MILD
   - MODERATE
   - SEVERE
5. Added state-transition hold logic.

Session Logger

1. Built system/session_logger.py.
2. Logged live frame data to CSV.
3. Saved session summaries to JSON.

Screen Controller

1. Built system/screen_control.py.
2. Added brightness and alert responses for fatigue levels.

8. Live Detector

1. Built main_detector.py.
2. Loaded the fatigue and eye models.
3. Opened the webcam.
4. Ran face analysis on each frame.
5. Ran CNN inference on detected face and eye regions.
6. Fused all scores into a final fatigue value.
7. Drew overlay text and confidence bars.
8. Added keyboard controls for quit, pause, reset, and screenshot.

9. Dashboard and Calibration

1. Built ui/dashboard.py for live visualization.
2. Built calibrate.py for user-specific calibration.
3. Added config/config.yaml for settings and thresholds.

10. Runtime Fixes Made During Testing

These are the practical fixes added after running the project live.

Startup and Camera Fixes

1. Fixed Windows console encoding issues in main_detector.py.
2. Added UTF-8 console output reconfiguration.
3. Added webcam probing across multiple indexes.
4. Added webcam probing across multiple backends:
   - default
   - MSMF
   - DSHOW
5. Verified camera availability with a separate webcam preview test.

Live Inference Fixes

1. Stopped feeding the full frame into the fatigue CNN.
2. Changed live fatigue inference to use the detected face crop only.
3. Changed live eye inference to use eye-region crops instead of the whole frame.
4. Reused the latest CNN outputs between inference frames so labels did not jump back to neutral.
5. Reset cached CNN values when no face was detected.

Model Output Fixes

1. Tested the saved models directly on validation images.
2. Discovered that the saved runtime outputs were reversed:
   - fatigue model produced higher values for non_fatigue
   - eye model produced higher values for eye_closed
3. Corrected the live runtime mapping:
   - fatigue probability is now 1 - raw_fatigue_output
   - eye-open probability is now 1 - raw_eye_output
4. Fixed OpenCV image color order before inference:
   - converted BGR to RGB

Fatigue Score Tuning

1. Made score recovery faster when the user returns to a normal expression.
2. Used asymmetric smoothing:
   - slower rise
   - faster fall
3. Used stricter fatigue boundaries so normal faces do not drift upward too easily.
4. Prevented zero blink history from being treated as immediate fatigue.

Face and Eye Box Stability

1. Added real face box output from FaceAnalyzer.
2. Added left-eye and right-eye tracking boxes.
3. Smoothed the face box over time.
4. Smoothed the eye boxes over time.
5. Added side-aware eye candidate selection.
6. Reused the previous eye box when eye detection briefly failed.
7. Added geometry-based estimated eye boxes from the face box.
8. Blended detector eye boxes with stable face-based eye templates.

Closed-Eye Detection Fixes

1. Added closed_eye_frames from the contour/EAR path.
2. Added cnn_closed_eye_frames from the eye CNN path.
3. Changed eye inference to score left and right eyes separately and average them.
4. Added sustained-closed-eye rules inside FatigueScorer.
5. Added on-screen debug values:
   - Closed Eye Frames
   - Eye CNN Closed
   - Face CNN
   - Eye CNN
   - Confidence

11. Main Files Used to Build the Project

- explore_dataset.py
- prepare_dataset.py
- train_fatigue_model.py
- train_eye_model.py
- evaluate_models.py
- core/face_analyzer.py
- core/fatigue_scorer.py
- system/session_logger.py
- system/screen_control.py
- main_detector.py
- ui/dashboard.py
- calibrate.py
- config/config.yaml
- run.py

12. Typical Build Order

1. Install dependencies.
2. Explore datasets.
3. Prepare and merge datasets.
4. Train fatigue model.
5. Train eye model.
6. Evaluate both models.
7. Build face analyzer.
8. Build fatigue scorer.
9. Build screen controller.
10. Build session logger.
11. Build live detector.
12. Build dashboard.
13. Add calibration.
14. Run live testing.
15. Fix runtime issues found during live testing.
16. Tune tracking, confidence display, and fatigue detection behavior.

13. Current Notes

1. ALL_STEPS.md remains the large implementation log.
2. info.md is the shorter practical build summary.
3. info.txt is the plain text version of the same summary.
4. The latest live detector includes the debugging fixes made during testing.
