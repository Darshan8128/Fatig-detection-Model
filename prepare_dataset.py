"""
════════════════════════════════════════════════════════════════
         STEP 2: DATASET PREPARATION & MERGING
════════════════════════════════════════════════════════════════
Auto-map subfolders, process images, balance classes, split dataset.
"""
import os
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import imagehash
import traceback

PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
PROCESSED_DIR = PROJECT_ROOT / "processed_dataset"
FINAL_DATASET_DIR = PROJECT_ROOT / "final_dataset"
CONFIG_DIR = PROJECT_ROOT / "config"
MAPPING_FILE = CONFIG_DIR / "dataset_mapping.json"
SPLIT_SUMMARY_FILE = CONFIG_DIR / "split_summary.json"

# Archive configurations
ARCHIVES = [
    ("archive (1)", "My Facial Dataset (Fatigue/NonFatigue)", "face"),
    ("archive (3)", "Driver Drowsiness Dataset DDD (Kaggle)", "face"),
    ("archive (4)", "Eye Dataset Open/Close (Kaggle)", "eye")
]

# Valid image extensions
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.JPG', '.JPEG', '.PNG'}

# Class mapping options
CLASS_OPTIONS = {
    '1': 'fatigue',
    '2': 'non_fatigue',
    '3': 'eye_open',
    '4': 'eye_closed',
    '5': 'skip'
}


class DatasetPreparer:
    """Prepares and merges datasets."""
    
    def __init__(self):
        self.mapping = {}  # folder -> target class
        self.image_hashes = {}  # hash -> filepath (for dedup)
        self.class_files = defaultdict(list)  # class -> list of files
        self.processed_count = 0
        self.skipped_count = 0
        self.corrupted_count = 0
    
    def log(self, message):
        """Print and log message."""
        # Replace emojis for Windows compatibility
        message = message.replace('\U0001f4c1', '[DIR]').replace('\u274c', '[X]').replace('\u2705', '[OK]')
        message = message.replace('\ud83d\udd0d', '[SEARCH]').replace('\ud83c\udf10', '[GLOBE]').replace('\u23f3', '[WAIT]')
        message = message.replace('\U0001f4be', '[SAVE]')
        print(message)
    
    def create_directories(self):
        """Create required directories."""
        self.log(f"\n[DIR] Creating directories...")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        for class_name in ['fatigue', 'non_fatigue', 'eye_open', 'eye_closed']:
            (PROCESSED_DIR / class_name).mkdir(exist_ok=True)
            (FINAL_DATASET_DIR / 'train' / class_name).mkdir(parents=True, exist_ok=True)
            (FINAL_DATASET_DIR / 'val' / class_name).mkdir(parents=True, exist_ok=True)
            (FINAL_DATASET_DIR / 'test' / class_name).mkdir(parents=True, exist_ok=True)
        
        self.log("[OK] Directories created.")
    
    def discover_subfolders(self):
        """Find all subfolders in archives."""
        self.log(f"\n[SEARCH] Discovering subfolders in archives...\n")
        subfolders = defaultdict(list)
        
        for archive_folder, archive_display, image_type in ARCHIVES:
            archive_path = DATASET_DIR / archive_folder
            
            if not archive_path.exists():
                self.log(f"⚠️  {archive_folder} not found.")
                continue
            
            self.log(f"[GLOBE] Archive: {archive_display}")
            
            # Find all directories with images
            for item in archive_path.rglob("*"):
                if not item.is_dir():
                    continue
                
                # Check if folder has images
                image_files = [f for f in item.iterdir() 
                              if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
                
                if image_files:
                    rel_path = item.relative_to(archive_path)
                    display_path = f"{archive_folder}/{rel_path}"
                    subfolders[archive_folder].append({
                        'path': item,
                        'rel_path': str(rel_path),
                        'display_path': display_path,
                        'count': len(image_files),
                        'image_type': image_type
                    })
        
        return subfolders
    
    def get_user_mapping(self, subfolders):
        """Interactively get user mapping for each subfolder."""
        self.log("\n" + "=" * 70)
        self.log("INTERACTIVE FOLDER MAPPING")
        self.log("=" * 70)
        self.log("Each subfolder will be mapped to a target class.\n")
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        mapping = {}
        
        for archive_folder in sorted(subfolders.keys()):
            folders_in_archive = subfolders[archive_folder]
            self.log(f"\n[PKG] {archive_folder}\n")
            
            for folder_info in folders_in_archive:
                display = folder_info['display_path']
                count = folder_info['count']
                image_type = folder_info['image_type']
                
                self.log(f"Found: {display}")
                self.log(f"  • Images: {count}")
                self.log(f"  • Type: {image_type} images")
                
                # Show options based on image type
                self.log("\nMap to:")
                if image_type == "face":
                    self.log("  [1] fatigue")
                    self.log("  [2] non_fatigue")
                    self.log("  [3] skip")
                else:  # eye
                    self.log("  [3] eye_open")
                    self.log("  [4] eye_closed")
                    self.log("  [5] skip")
                
                while True:
                    choice = input("\nYour choice (1-5): ").strip()
                    
                    if choice in CLASS_OPTIONS:
                        target_class = CLASS_OPTIONS[choice]
                        if target_class == 'skip':
                            self.log(f"↷ SKIPPED\n")
                            mapping[folder_info['rel_path']] = 'skip'
                        else:
                            self.log(f"✅ → {target_class}\n")
                            mapping[folder_info['rel_path']] = target_class
                        break
                    else:
                        self.log("❌ Invalid choice. Try again.")
        
        return mapping
    
    def save_mapping(self, mapping):
        """Save mapping to JSON."""
        self.log(f"\n[SAVE] Saving mapping to {MAPPING_FILE}...")
        with open(MAPPING_FILE, 'w') as f:
            json.dump(mapping, f, indent=2)
        self.log("[OK] Mapping saved.")
    
    def rotate_image(self, image, angle):
        """Rotate image by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def augment_image(self, image_cv):
        """Apply random augmentations."""
        augmented = []
        
        # Original
        augmented.append(image_cv.copy())
        
        # Flipped
        augmented.append(cv2.flip(image_cv, 1))
        
        # Rotated
        for angle in [-15, 15]:
            augmented.append(self.rotate_image(image_cv, angle))
        
        # Brightness adjusted
        for brightness in [0.8, 1.2]:
            aug = cv2.convertScaleAbs(image_cv, alpha=brightness, beta=0)
            augmented.append(aug)
        
        return augmented
    
    def apply_clahe(self, image_cv):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if len(image_cv.shape) == 3:
            # Convert to LAB and apply to L channel
            lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image_cv)
    
    def process_image(self, source_file, target_size, target_class):
        """Process single image: resize, convert, deduplicate."""
        try:
            # Open with PIL for validation
            img_pil = Image.open(source_file)
            img_pil.verify()
            
            # Open with OpenCV for processing
            img_cv = cv2.imread(str(source_file))
            if img_cv is None:
                return None, None
            
            # Convert grayscale to RGB
            if len(img_cv.shape) == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            
            # Resize
            img_cv = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_AREA)
            
            # Apply CLAHE if image looks dark
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            if gray.mean() < 100:  # Dark image
                img_cv = self.apply_clahe(img_cv)
            
            # Calculate hash for deduplication
            img_hash = imagehash.phash(Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)))
            
            return img_cv, str(img_hash)
        
        except Exception as e:
            return None, None
    
    def process_and_merge_datasets(self, subfolders, mapping):
        """Process all images and merge into processed_dataset."""
        self.log("\n" + "=" * 70)
        self.log("[PROCESS] PROCESSING IMAGES")
        self.log("=" * 70)
        
        face_size = (224, 224)
        eye_size = (64, 64)
        
        for archive_folder in sorted(subfolders.keys()):
            folders_in_archive = subfolders[archive_folder]
            
            for folder_info in folders_in_archive:
                folder_path = folder_info['path']
                rel_path = folder_info['rel_path']
                
                # Skip if not mapped
                if rel_path not in mapping or mapping[rel_path] == 'skip':
                    continue
                
                target_class = mapping[rel_path]
                image_type = folder_info['image_type']
                target_size = eye_size if image_type == "eye" else face_size
                
                self.log(f"\n[INPUT] Processing: {rel_path}")
                self.log(f"   [->] Class: {target_class}")
                
                # Get all images in folder
                image_files = sorted([f for f in folder_path.iterdir()
                                     if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])
                
                valid_count = 0
                for img_file in image_files:
                    img_cv, img_hash = self.process_image(img_file, target_size, target_class)
                    
                    if img_cv is None:
                        self.corrupted_count += 1
                        continue
                    
                    # Check for duplicates
                    if img_hash in self.image_hashes:
                        self.skipped_count += 1
                        continue
                    
                    self.image_hashes[img_hash] = img_file
                    
                    # Save processed image
                    output_file = PROCESSED_DIR / target_class / f"{target_class}_{self.processed_count:05d}.jpg"
                    cv2.imwrite(str(output_file), img_cv)
                    self.class_files[target_class].append(output_file)
                    
                    self.processed_count += 1
                    valid_count += 1
                
                self.log(f"   [OK] {valid_count} images processed")
        
        self.log(f"\n{'=' * 70}")
        self.log(f"Processing complete:")
        self.log(f"  [*] Processed: {self.processed_count}")
        self.log(f"  [*] Duplicates: {self.skipped_count}")
        self.log(f"  [*] Corrupted: {self.corrupted_count}")
        self.log(f"{'=' * 70}\n")
    
    def balance_classes(self):
        """Balance class distribution via augmentation."""
        self.log("\n" + "=" * 70)
        self.log("[BALANCE] BALANCING CLASSES")
        self.log("=" * 70)
        
        # Check imbalance
        class_counts = {cls: len(files) for cls, files in self.class_files.items()}
        
        if not class_counts:
            self.log("❌ No classes found!")
            return
        
        self.log(f"\nCurrent class distribution:")
        for cls, count in sorted(class_counts.items()):
            self.log(f"  [*] {cls:15s}: {count:5d} images")
        
        max_count = max(class_counts.values())
        
        # Check if rebalancing needed
        needs_balance = any(count < max_count * 0.5 for count in class_counts.values())
        
        if needs_balance:
            self.log("\n[WARN] Class imbalance detected. Augmenting minority classes...\n")
            
            for target_class, files in self.class_files.items():
                current_count = len(files)
                target_count = max_count
                augment_needed = target_count - current_count
                
                if augment_needed > 0:
                    self.log(f"[AUG] Augmenting {target_class}: {current_count} -> {target_count}")
                    
                    augmented = 0
                    file_idx = 0
                    
                    while augmented < augment_needed and file_idx < len(files):
                        source_file = files[file_idx]
                        
                        try:
                            img_cv = cv2.imread(str(source_file))
                            if img_cv is None:
                                file_idx += 1
                                continue
                            
                            augmentations = self.augment_image(img_cv)
                            
                            for aug_img in augmentations[1:]:  # Skip original
                                if augmented >= augment_needed:
                                    break
                                
                                output_file = PROCESSED_DIR / target_class / \
                                            f"{target_class}_aug_{self.processed_count:05d}.jpg"
                                cv2.imwrite(str(output_file), aug_img)
                                
                                self.class_files[target_class].append(output_file)
                                self.processed_count += 1
                                augmented += 1
                        
                        except:
                            pass
                        
                        file_idx += 1
                    
                    self.log(f"   [OK] Added {augmented} augmented images\n")
        else:
            self.log("\n[OK] Classes are well balanced. No augmentation needed.\n")
    
    def split_dataset(self):
        """Split into train/val/test (80/10/10)."""
        self.log("\n" + "=" * 70)
        self.log("[SPLIT] SPLITTING DATASET")
        self.log("=" * 70)
        
        split_info = {
            'train': {},
            'val': {},
            'test': {},
            'total': {},
            'split_ratio': {'train': 0.8, 'val': 0.1, 'test': 0.1}
        }
        
        for target_class, files in self.class_files.items():
            random.shuffle(files)
            total = len(files)
            
            train_count = int(total * 0.8)
            val_count = int(total * 0.1)
            
            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]
            
            self.log(f"\n{target_class}:")
            self.log(f"  [*] Total: {total}")
            self.log(f"  [*] Train (80%): {len(train_files)}")
            self.log(f"  [*] Val (10%): {len(val_files)}")
            self.log(f"  [*] Test (10%): {len(test_files)}")
            
            split_info['total'][target_class] = total
            split_info['train'][target_class] = len(train_files)
            split_info['val'][target_class] = len(val_files)
            split_info['test'][target_class] = len(test_files)
            
            # Copy files
            for src_file in train_files:
                dst_file = FINAL_DATASET_DIR / 'train' / target_class / src_file.name
                shutil.copy2(src_file, dst_file)
            
            for src_file in val_files:
                dst_file = FINAL_DATASET_DIR / 'val' / target_class / src_file.name
                shutil.copy2(src_file, dst_file)
            
            for src_file in test_files:
                dst_file = FINAL_DATASET_DIR / 'test' / target_class / src_file.name
                shutil.copy2(src_file, dst_file)
        
        # Save split summary
        self.log(f"\n[SAVE] Saving split summary to {SPLIT_SUMMARY_FILE}...")
        with open(SPLIT_SUMMARY_FILE, 'w') as f:
            json.dump(split_info, f, indent=2)
        self.log("[OK] Split summary saved.")
        
        return split_info
    
    def run(self):
        """Main pipeline."""
        self.log("\n" + "=" * 70)
        self.log("DATASET PREPARATION & MERGING")
        self.log("=" * 70)
        
        # Step 1: Create directories
        self.create_directories()
        
        # Step 2: Discover subfolders
        subfolders = self.discover_subfolders()
        
        if not subfolders:
            self.log("[ERROR] No image subfolders found!")
            return False
        
        # Step 3: Get user mapping
        mapping = self.get_user_mapping(subfolders)
        self.save_mapping(mapping)
        
        # Step 4: Process and merge
        self.process_and_merge_datasets(subfolders, mapping)
        
        if self.processed_count == 0:
            self.log("[ERROR] No images were processed!")
            return False
        
        # Step 5: Balance classes
        self.balance_classes()
        
        # Step 6: Split dataset
        split_info = self.split_dataset()
        
        # Final summary
        self.log("\n" + "=" * 70)
        self.log("[OK] DATASET PREPARATION COMPLETE")
        self.log("=" * 70)
        self.log(f"\n[OK] Dataset ready in: {FINAL_DATASET_DIR}")
        self.log(f"\nNext steps:")
        self.log(f"  [1] STEP 3: Train Fatigue Model")
        self.log(f"  [2] STEP 4: Train Eye Model")
        self.log(f"\nFiles created:")
        self.log(f"  [*] {MAPPING_FILE}")
        self.log(f"  [*] {SPLIT_SUMMARY_FILE}")
        self.log(f"  [*] final_dataset/train/val/test/ directories\n")
        
        return True


def main():
    """Main entry point."""
    try:
        preparer = DatasetPreparer()
        preparer.run()
        print("\n✅ [prepare_dataset.py] complete. Type NEXT for next file.\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
