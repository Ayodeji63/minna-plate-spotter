#!/usr/bin/env python3
"""
Nigerian Vehicle Plate Number Recognition System - ENHANCED VERSION
With Camera Display and Better Detection
"""

import cv2
import pytesseract
import numpy as np
import re
import logging
from datetime import datetime
import os
import json
from pathlib import Path
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("Warning: PiCamera2 not available. Will use USB camera fallback.")

class NigerianPlateRecognizer:
    def __init__(self, config_path="config.json"):
        """Initialize the Nigerian Plate Recognition System"""
        self.setup_logging()
        self.load_config(config_path)
        self.setup_camera()
        self.setup_directories()
        
        # Enhanced Nigerian plate patterns for 8-character plates
        self.nigerian_patterns = [
            r'^[A-Z]{3}[-\s]?\d{3}[-\s]?[A-Z]{2}$',     # ABC-123-DE (8 chars) - like GGE123ZY
            r'^[A-Z]{2}[-\s]?\d{4}[-\s]?[A-Z]{2}$',     # AB-1234-CD (8 chars)
            r'^[A-Z]{3}[-\s]?\d{2}[-\s]?[A-Z]{3}$',     # ABC-12-DEF (8 chars)
            r'^[A-Z]{4}[-\s]?\d{3}[-\s]?[A-Z]{1}$',     # ABCD-123-E (8 chars)
            r'^[A-Z]{1}[-\s]?\d{3}[-\s]?[A-Z]{4}$',     # A-123-BCDE (8 chars)
            r'^[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{3}$',     # AB-123-CDE (8 chars)
            r'^[A-Z]{3}[-\s]?\d{4}[-\s]?[A-Z]{1}$',     # ABC-1234-D (8 chars)
            # More flexible pattern
            r'^[A-Z]{2,4}[-\s]?\d{2,4}[-\s]?[A-Z]{1,4}$'
        ]
        
        # Enhanced state codes
        self.niger_state_codes = [
            'NGR', 'MIN', 'BID', 'SUL', 'KON',  # Niger state
            'APP', 'LAG', 'ABJ', 'KAN', 'OYO',  # Common Nigerian states
            'EDO', 'RIV', 'ANA', 'IMO', 'ABI',
            'GGE', 'FCT', 'KD', 'KB', 'KT'      # Added more including GGE
        ]
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('plate_recognition.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path):
        """Load configuration with better defaults"""
        default_config = {
            "camera": {
                "resolution": [1920, 1080],
                "framerate": 30,
                "rotation": 0,
                "brightness": 50,
                "contrast": 50
            },
            "detection": {
                "min_plate_width": 60,       # Even smaller minimum
                "min_plate_height": 20,      # Even smaller minimum
                "max_plate_width": 600,      # Larger maximum
                "max_plate_height": 250,     # Larger maximum
                "confidence_threshold": 0.4,  # Lower threshold
                "aspect_ratio_min": 1.2,     # More flexible
                "aspect_ratio_max": 7.0,     # More flexible
                "min_area": 300              # Much smaller minimum area
            },
            "ocr": {
                "psm": 8,
                "oem": 3,
                "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "dpi": 300
            },
            "preprocessing": {
                "gaussian_blur_kernel": 3,
                "adaptive_threshold_block_size": 11,
                "adaptive_threshold_c": 2,
                "morphology_kernel_size": [2, 2],
                "min_resize_height": 40,
                "contrast_alpha": 1.8,      # Higher contrast
                "brightness_beta": 20       # More brightness adjustment
            },
            "display": {
                "show_camera": True,
                "window_width": 1024,
                "window_height": 768,
                "show_processed": True
            },
            "storage": {
                "save_images": True,
                "save_detections": True,
                "max_stored_images": 1000
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config = self.deep_merge(default_config, loaded_config)
            else:
                self.config = default_config
                self.save_config(config_path)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            self.config = default_config
            
    def deep_merge(self, default, override):
        """Deep merge dictionaries"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        return result
            
    def save_config(self, config_path):
        """Save configuration"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            
    def setup_camera(self):
        """Initialize camera"""
        try:
            if PI_CAMERA_AVAILABLE:
                self.camera = Picamera2()
                camera_config = self.camera.create_still_configuration(
                    main={"size": tuple(self.config["camera"]["resolution"])}
                )
                self.camera.configure(camera_config)
                self.camera.start()
                self.camera_type = "picamera"
                self.logger.info("PiCamera initialized successfully")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["resolution"][0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["resolution"][1])
                self.camera_type = "usb"
                self.logger.info("USB Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
            
    def setup_directories(self):
        """Create directories"""
        directories = ['captured_plates', 'detected_plates', 'logs', 'data', 'debug_images']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def capture_frame(self):
        """Capture frame from camera"""
        try:
            if self.camera_type == "picamera":
                frame = self.camera.capture_array()
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.camera.read()
                if not ret:
                    return None
            return frame
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None
            
    def preprocess_image(self, image, debug=False):
        """Enhanced preprocessing with multiple approaches"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Standard preprocessing
        prep_config = self.config["preprocessing"]
        alpha = prep_config.get("contrast_alpha", 1.8)
        beta = prep_config.get("brightness_beta", 20)
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply different preprocessing methods
        methods = []
        
        # Method 1: Adaptive threshold
        blurred1 = cv2.GaussianBlur(enhanced, (5, 5), 0)
        thresh1 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append(("adaptive_gaussian", thresh1))
        
        # Method 2: OTSU threshold
        blurred2 = cv2.GaussianBlur(enhanced, (5, 5), 0)
        _, thresh2 = cv2.threshold(blurred2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("otsu", thresh2))
        
        # Method 3: Adaptive threshold with mean
        thresh3 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 10)
        methods.append(("adaptive_mean", thresh3))
        
        # Method 4: Simple binary threshold
        _, thresh4 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        methods.append(("simple_binary", thresh4))
        
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"debug_images/{timestamp}_0_original.jpg", gray)
            cv2.imwrite(f"debug_images/{timestamp}_1_enhanced.jpg", enhanced)
            for i, (name, method_img) in enumerate(methods):
                cv2.imwrite(f"debug_images/{timestamp}_2_{i}_{name}.jpg", method_img)
        
        return methods
        
    def detect_plate_regions_multi_method(self, image, debug=False):
        """Detect plates using multiple preprocessing methods"""
        all_candidates = []
        methods = self.preprocess_image(image, debug)
        
        detection_config = self.config["detection"]
        
        for method_name, processed in methods:
            # Find contours for this method
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            method_candidates = []
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Check criteria
                width_ok = detection_config["min_plate_width"] <= w <= detection_config["max_plate_width"]
                height_ok = detection_config["min_plate_height"] <= h <= detection_config["max_plate_height"]
                ratio_ok = detection_config["aspect_ratio_min"] <= aspect_ratio <= detection_config["aspect_ratio_max"]
                area_ok = area > detection_config["min_area"]
                
                if width_ok and height_ok and ratio_ok and area_ok:
                    # Extract region with padding
                    padding = 10
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    plate_region = image[y1:y2, x1:x2]
                    
                    if plate_region.size > 0:
                        candidate = {
                            'region': plate_region,
                            'bbox': (x1, y1, x2-x1, y2-y1),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'method': method_name,
                            'contour_index': i
                        }
                        method_candidates.append(candidate)
                        
                        if debug:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(f"debug_images/{timestamp}_{method_name}_candidate_{i}.jpg", plate_region)
            
            all_candidates.extend(method_candidates)
            self.logger.info(f"Method '{method_name}' found {len(method_candidates)} candidates")
        
        # Remove duplicates and sort by area
        unique_candidates = self.remove_duplicate_candidates(all_candidates)
        unique_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        self.logger.info(f"Total unique candidates: {len(unique_candidates)}")
        return unique_candidates
        
    def remove_duplicate_candidates(self, candidates):
        """Remove duplicate candidates based on position overlap"""
        unique = []
        for candidate in candidates:
            x1, y1, w1, h1 = candidate['bbox']
            is_duplicate = False
            
            for existing in unique:
                x2, y2, w2, h2 = existing['bbox']
                
                # Check for significant overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                min_area = min(area1, area2)
                
                if overlap_area > 0.5 * min_area:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(candidate)
        
        return unique
        
    def enhance_plate_for_ocr(self, plate_image, debug=False):
        """Multiple enhancement approaches for OCR"""
        if plate_image.size == 0:
            return []
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        enhanced_versions = []
        
        # Version 1: High contrast
        alpha_high = 2.5
        beta_high = 30
        high_contrast = cv2.convertScaleAbs(gray, alpha=alpha_high, beta=beta_high)
        
        # Resize for OCR
        height, width = high_contrast.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            high_contrast = cv2.resize(high_contrast, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Apply different thresholding
        _, thresh1 = cv2.threshold(high_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(high_contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        enhanced_versions.append(("high_contrast_binary", thresh1))
        enhanced_versions.append(("high_contrast_binary_inv", thresh2))
        
        # Version 2: Histogram equalization
        equalized = cv2.equalizeHist(gray)
        if equalized.shape[0] < 50:
            scale = 50 / equalized.shape[0]
            new_width = int(equalized.shape[1] * scale)
            equalized = cv2.resize(equalized, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        _, eq_thresh1 = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, eq_thresh2 = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        enhanced_versions.append(("equalized_binary", eq_thresh1))
        enhanced_versions.append(("equalized_binary_inv", eq_thresh2))
        
        # Version 3: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_close = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        morph_open = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)
        
        enhanced_versions.append(("morphological", morph_open))
        
        if debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, (name, img) in enumerate(enhanced_versions):
                cv2.imwrite(f"debug_images/{timestamp}_ocr_prep_{i}_{name}.jpg", img)
        
        return enhanced_versions
        
    def extract_text_comprehensive(self, plate_image, debug=False):
        """Comprehensive text extraction using multiple methods"""
        enhanced_versions = self.enhance_plate_for_ocr(plate_image, debug)
        
        ocr_config = self.config["ocr"]
        psm_modes = [8, 7, 13, 6, 3]  # Multiple PSM modes
        
        all_results = []
        
        for version_name, enhanced_img in enhanced_versions:
            for psm in psm_modes:
                try:
                    custom_config = f'--oem {ocr_config["oem"]} --psm {psm} -c tessedit_char_whitelist={ocr_config["whitelist"]}'
                    
                    # Get detailed data
                    data = pytesseract.image_to_data(enhanced_img, config=custom_config, output_type=pytesseract.Output.DICT)
                    
                    # Extract confident text
                    words = []
                    confidences = []
                    
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 20:  # Lower confidence threshold
                            text = data['text'][i].strip().upper()
                            if text and len(text) > 0:
                                words.append(text)
                                confidences.append(int(conf))
                    
                    if words:
                        # Try different combinations
                        full_text = ''.join(words)
                        spaced_text = ' '.join(words)
                        
                        for raw_text in [full_text, spaced_text]:
                            cleaned = self.clean_plate_text(raw_text)
                            if len(cleaned) >= 5:  # Minimum length
                                avg_conf = np.mean(confidences) if confidences else 0
                                all_results.append({
                                    'text': cleaned,
                                    'confidence': avg_conf,
                                    'method': f"{version_name}_psm{psm}",
                                    'raw_text': raw_text
                                })
                
                except Exception as e:
                    if debug:
                        self.logger.debug(f"OCR failed for {version_name} PSM {psm}: {e}")
                    continue
        
        # Sort by confidence and return best results
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        if debug and all_results:
            self.logger.info("OCR Results:")
            for i, result in enumerate(all_results[:5]):  # Show top 5
                self.logger.info(f"  {i+1}. '{result['text']}' (conf: {result['confidence']:.1f}, method: {result['method']})")
        
        return all_results
        
    def clean_plate_text(self, text):
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper().strip())
        cleaned = re.sub(r'\s+', '', cleaned)  # Remove all spaces
        
        # Length check
        if len(cleaned) < 5:
            return cleaned
        
        # Apply corrections
        corrected = list(cleaned)
        
        # First 3 characters - likely letters
        for i in range(min(3, len(corrected))):
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
                elif corrected[i] == '5': corrected[i] = 'S'
                elif corrected[i] == '8': corrected[i] = 'B'
                elif corrected[i] == '6': corrected[i] = 'G'
        
        # Middle characters (positions 3-6) - likely numbers
        for i in range(3, min(6, len(corrected))):
            if corrected[i].isalpha():
                if corrected[i] == 'O': corrected[i] = '0'
                elif corrected[i] in 'IL': corrected[i] = '1'
                elif corrected[i] == 'S': corrected[i] = '5'
                elif corrected[i] == 'B': corrected[i] = '8'
                elif corrected[i] == 'G': corrected[i] = '6'
        
        # Last 2 characters - likely letters
        for i in range(max(0, len(corrected)-2), len(corrected)):
            if corrected[i].isdigit():
                if corrected[i] == '0': corrected[i] = 'O'
                elif corrected[i] == '1': corrected[i] = 'I'
                elif corrected[i] == '5': corrected[i] = 'S'
                elif corrected[i] == '8': corrected[i] = 'B'
                elif corrected[i] == '6': corrected[i] = 'G'
        
        return ''.join(corrected)
        
    def validate_nigerian_plate(self, plate_text):
        """Enhanced validation"""
        if not plate_text or len(plate_text) < 5:
            return False, 0.0
        
        normalized = re.sub(r'[-\s]', '', plate_text.upper())
        
        confidence_scores = []
        
        # Test against patterns
        for pattern in self.nigerian_patterns:
            if re.match(pattern, normalized):
                base_score = 0.8
                
                # Length bonus (8 chars is ideal)
                if len(normalized) == 8:
                    length_bonus = 0.2
                elif 7 <= len(normalized) <= 9:
                    length_bonus = 0.1
                else:
                    length_bonus = 0.0
                
                # State code bonus
                state_bonus = 0.0
                for code in self.niger_state_codes:
                    if code in normalized:
                        state_bonus = 0.1
                        break
                
                # Character distribution
                letters = sum(1 for c in normalized if c.isalpha())
                digits = sum(1 for c in normalized if c.isdigit())
                
                if 3 <= letters <= 5 and 2 <= digits <= 5:
                    dist_bonus = 0.1
                else:
                    dist_bonus = 0.0
                
                total_score = base_score + length_bonus + state_bonus + dist_bonus
                confidence_scores.append(min(1.0, total_score))
        
        if confidence_scores:
            max_confidence = max(confidence_scores)
            threshold = self.config["detection"]["confidence_threshold"]
            return max_confidence >= threshold, max_confidence
        
        return False, 0.0
        
    def save_detection(self, image, plate_text, bbox, confidence):
        """Save detection with enhanced info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config["storage"]["save_images"]:
            x, y, w, h = bbox
            detection_image = image.copy()
            cv2.rectangle(detection_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(detection_image, f"{plate_text} ({confidence:.2f})", 
                       (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            image_path = f"detected_plates/{timestamp}_{plate_text}_{confidence:.2f}.jpg"
            cv2.imwrite(image_path, detection_image)
            
        if self.config["storage"]["save_detections"]:
            detection_data = {
                'timestamp': timestamp,
                'plate_number': plate_text,
                'confidence': confidence,
                'bbox': bbox,
                'location': 'Minna, Nigeria',
                'character_count': len(plate_text)
            }
            
            data_path = f"data/{timestamp}_{plate_text}_{confidence:.2f}.json"
            with open(data_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
                
        self.logger.info(f"🎯 PLATE DETECTED: {plate_text} (Confidence: {confidence:.2f})")
        return True
        
    def process_frame(self, image, debug=False):
        """Enhanced frame processing"""
        if image is None:
            return []
        
        detections = []
        
        # Get candidates using multiple methods
        candidates = self.detect_plate_regions_multi_method(image, debug)
        
        self.logger.info(f"Processing {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates[:8]):  # Process more candidates
            self.logger.info(f"Processing candidate {i+1}: {candidate['method']} (area: {candidate['area']:.0f})")
            
            plate_region = candidate['region']
            bbox = candidate['bbox']
            
            # Get all OCR results
            ocr_results = self.extract_text_comprehensive(plate_region, debug)
            
            # Try each result
            for result in ocr_results[:3]:  # Try top 3 results
                plate_text = result['text']
                ocr_confidence = result['confidence']
                
                if plate_text and len(plate_text) >= 5:
                    self.logger.info(f"  Testing: '{plate_text}' (OCR conf: {ocr_confidence:.1f})")
                    
                    is_valid, validation_confidence = self.validate_nigerian_plate(plate_text)
                    
                    if is_valid:
                        final_confidence = (ocr_confidence / 100.0 * 0.6) + (validation_confidence * 0.4)
                        
                        self.save_detection(image, plate_text, bbox, final_confidence)
                        
                        detections.append({
                            'plate_number': plate_text,
                            'confidence': final_confidence,
                            'bbox': bbox,
                            'method': candidate['method'],
                            'ocr_method': result['method'],
                            'timestamp': datetime.now()
                        })
                        break  # Found valid plate, move to next candidate
                    else:
                        self.logger.info(f"    Validation failed (conf: {validation_confidence:.2f})")
        
        return detections
        
    def run_continuous_detection(self, debug=False):
        """Continuous detection with camera display"""
        self.logger.info("🚀 Starting continuous detection with camera display...")
        try:
            frame_count = 0
            while True:
                frame_count += 1
                # Capture frame
                frame = self.capture_frame()
                if frame is not None:
                    # Create display frame
                    display_frame = frame.copy()
                    # Process for detection
                    detections = self.process_frame(frame, debug)
                    # Draw detections on display
                    for detection in detections:
                        x, y, w, h = detection['bbox']
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        # Draw label
                        label = f"{detection['plate_number']} ({detection['confidence']:.2f})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(display_frame, (x, y-30), (x + label_size[0], y), (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    # Add frame info
                    info_text = f"Frame: {frame_count} | Detections: {len(detections)}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # Resize for display
                    display_config = self.config.get("display", {})
                    if display_config.get("show_camera", True):
                        display_width = display_config.get("window_width", 1024)
                        display_height = display_config.get("window_height", 768)
                        display_frame = cv2.resize(display_frame, (display_width, display_height))
                        cv2.imshow("Nigerian Plate Recognition", display_frame)
                    # Optionally show processed images (for debug)
                    if debug and display_config.get("show_processed", True):
                        processed_methods = self.preprocess_image(frame, debug=False)
                        for i, (name, img) in enumerate(processed_methods):
                            win_name = f"Processed: {name}"
                            cv2.imshow(win_name, cv2.resize(img, (400, 300)))
                # Wait for key event, allow quitting with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quitting continuous detection (q pressed)")
                    break
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
        finally:
            cv2.destroyAllWindows()
            self.logger.info("🧹 Cleanup completed")
# Add main entry point
def main():
    recognizer = NigerianPlateRecognizer()
    print("🇳🇬 Nigerian Vehicle Plate Recognition System - ENHANCED VERSION")
    print("=" * 60)
    print("1. Continuous Detection")
    print("2. Continuous Detection (Debug)")
    choice = input("Select mode (1-2): ").strip()
    if choice == "1":
        recognizer.run_continuous_detection(debug=False)
    elif choice == "2":
        recognizer.run_continuous_detection(debug=True)
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()