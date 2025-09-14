#!/usr/bin/env python3
"""
Nigerian Vehicle Plate Number Recognition System
Designed for Raspberry Pi with Pi Camera
Location: Minna, Nigeria

Author: Vehicle Recognition System
Date: 2025

Dependencies:
pip install opencv-python pytesseract picamera2 numpy pillow
sudo apt-get install tesseract-ocr tesseract-ocr-eng
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
        
        # Nigerian plate patterns
        self.nigerian_patterns = [
            r'^[A-Z]{3}[-\s]?\d{3}[-\s]?[A-Z]{2}$',  # ABC-123-DE format
            r'^[A-Z]{2}[-\s]?\d{3}[-\s]?[A-Z]{3}$',  # AB-123-CDE format
            r'^[A-Z]{3}[-\s]?\d{2}[-\s]?[A-Z]{2}$',   # ABC-12-DE format (older)
            r'^\d{2}[-\s]?[A-Z]{2}[-\s]?\d{3}$',      # 12-AB-123 format
        ]
        
        # Minna/Niger state specific patterns
        self.niger_state_codes = ['NGR', 'MIN', 'BID', 'SUL', 'KON']
        
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
        """Load configuration from JSON file"""
        default_config = {
            "camera": {
                "resolution": [1920, 1080],
                "framerate": 30,
                "rotation": 0
            },
            "detection": {
                "min_plate_width": 100,
                "min_plate_height": 30,
                "max_plate_width": 400,
                "max_plate_height": 150,
                "confidence_threshold": 0.6
            },
            "ocr": {
                "psm": 8,  # Tesseract Page Segmentation Mode
                "oem": 3,  # OCR Engine Mode
                "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
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
                    # Merge with defaults
                    self.config = {**default_config, **loaded_config}
            else:
                self.config = default_config
                self.save_config(config_path)
        except Exception as e:
            self.logger.warning(f"Error loading config: {e}. Using defaults.")
            self.config = default_config
            
    def save_config(self, config_path):
        """Save current configuration to JSON file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            
    def setup_camera(self):
        """Initialize camera based on availability"""
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
                # Fallback to USB camera
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["resolution"][0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["resolution"][1])
                self.camera_type = "usb"
                self.logger.info("USB Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            raise
            
    def setup_directories(self):
        """Create necessary directories for storing data"""
        directories = ['captured_plates', 'detected_plates', 'logs', 'data']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def capture_frame(self):
        """Capture a frame from the camera"""
        try:
            if self.camera_type == "picamera":
                # Capture from PiCamera
                frame = self.camera.capture_array()
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # Capture from USB camera
                ret, frame = self.camera.read()
                if not ret:
                    return None
            return frame
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None
            
    def preprocess_image(self, image):
        """Preprocess image for better plate detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
        
    def detect_plate_regions(self, image):
        """Detect potential license plate regions"""
        processed = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_candidates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if dimensions match typical plate ratios
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Nigerian plates typically have aspect ratio between 2:1 and 5:1
            if (2.0 <= aspect_ratio <= 5.0 and 
                self.config["detection"]["min_plate_width"] <= w <= self.config["detection"]["max_plate_width"] and
                self.config["detection"]["min_plate_height"] <= h <= self.config["detection"]["max_plate_height"] and
                area > 1000):
                
                # Extract region of interest
                plate_region = image[y:y+h, x:x+w]
                plate_candidates.append({
                    'region': plate_region,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort by area (larger plates first)
        plate_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        return plate_candidates
        
    def enhance_plate_image(self, plate_image):
        """Enhance plate image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(equalized, 11, 17, 17)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize for better OCR (minimum height of 50 pixels)
        height, width = thresh.shape
        if height < 50:
            scale_factor = 50 / height
            new_width = int(width * scale_factor)
            thresh = cv2.resize(thresh, (new_width, 50), interpolation=cv2.INTER_CUBIC)
            
        return thresh
        
    def extract_text_from_plate(self, plate_image):
        """Extract text from plate using Tesseract OCR"""
        enhanced_plate = self.enhance_plate_image(plate_image)
        
        # Configure Tesseract
        custom_config = f'--oem {self.config["ocr"]["oem"]} --psm {self.config["ocr"]["psm"]} -c tessedit_char_whitelist={self.config["ocr"]["whitelist"]}'
        
        try:
            # Extract text
            text = pytesseract.image_to_string(enhanced_plate, config=custom_config)
            
            # Clean and normalize text
            cleaned_text = self.clean_plate_text(text)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""
            
    def clean_plate_text(self, text):
        """Clean and normalize extracted plate text"""
        # Remove whitespace and convert to uppercase
        cleaned = text.strip().upper()
        
        # Remove special characters except hyphens
        cleaned = re.sub(r'[^A-Z0-9\-\s]', '', cleaned)
        
        # Normalize spacing
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Common OCR corrections for Nigerian context
        corrections = {
            '0': 'O',  # Context-dependent
            'I': '1',  # Context-dependent
            '5': 'S',  # Context-dependent
            '8': 'B',  # Context-dependent
        }
        
        # Apply corrections based on position and context
        cleaned = self.apply_contextual_corrections(cleaned)
        
        return cleaned.strip()
        
    def apply_contextual_corrections(self, text):
        """Apply contextual corrections based on Nigerian plate patterns"""
        # Remove spaces and hyphens for pattern matching
        normalized = re.sub(r'[-\s]', '', text)
        
        if len(normalized) >= 6:  # Minimum viable plate length
            # Apply position-based corrections
            corrected = list(normalized)
            
            # First 2-3 characters should be letters
            for i in range(min(3, len(corrected))):
                if corrected[i] in '0123456789':
                    if corrected[i] == '0':
                        corrected[i] = 'O'
                    elif corrected[i] == '1':
                        corrected[i] = 'I'
                    elif corrected[i] == '5':
                        corrected[i] = 'S'
                        
            # Middle characters are typically numbers
            middle_start = 3 if len(corrected) > 6 else 2
            middle_end = len(corrected) - 2
            
            for i in range(middle_start, middle_end):
                if corrected[i] in 'OSHIT':
                    if corrected[i] == 'O':
                        corrected[i] = '0'
                    elif corrected[i] == 'S':
                        corrected[i] = '5'
                    elif corrected[i] in 'HIT':
                        corrected[i] = '1'
                        
            return ''.join(corrected)
            
        return text
        
    def validate_nigerian_plate(self, plate_text):
        """Validate if text matches Nigerian plate patterns"""
        if not plate_text or len(plate_text) < 5:
            return False, 0.0
            
        # Remove spaces and hyphens for pattern matching
        normalized = re.sub(r'[-\s]', '', plate_text)
        
        confidence_scores = []
        
        for pattern in self.nigerian_patterns:
            # Test with different formatting
            test_formats = [
                normalized,
                f"{normalized[:3]}-{normalized[3:6]}-{normalized[6:]}",
                f"{normalized[:2]} {normalized[2:5]} {normalized[5:]}",
            ]
            
            for test_format in test_formats:
                if re.match(pattern, test_format):
                    # Calculate confidence based on pattern match and length
                    base_confidence = 0.8
                    length_bonus = min(0.2, (len(normalized) - 5) * 0.05)
                    confidence = base_confidence + length_bonus
                    confidence_scores.append(confidence)
                    
        if confidence_scores:
            max_confidence = max(confidence_scores)
            return max_confidence >= self.config["detection"]["confidence_threshold"], max_confidence
            
        return False, 0.0
        
    def save_detection(self, image, plate_text, bbox, confidence):
        """Save detected plate information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config["storage"]["save_images"]:
            # Save original image with bounding box
            x, y, w, h = bbox
            detection_image = image.copy()
            cv2.rectangle(detection_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(detection_image, plate_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            image_path = f"detected_plates/{timestamp}_{plate_text}.jpg"
            cv2.imwrite(image_path, detection_image)
            
        if self.config["storage"]["save_detections"]:
            # Save detection data
            detection_data = {
                'timestamp': timestamp,
                'plate_number': plate_text,
                'confidence': confidence,
                'bbox': bbox,
                'location': 'Minna, Nigeria'
            }
            
            data_path = f"data/{timestamp}_{plate_text}.json"
            with open(data_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
                
        self.logger.info(f"Plate detected: {plate_text} (Confidence: {confidence:.2f})")
        
    def process_frame(self, image):
        """Process a single frame for plate detection"""
        if image is None:
            return []
        
        detections = []
        
        # Detect potential plate regions
        plate_candidates = self.detect_plate_regions(image)
        
        for candidate in plate_candidates[:3]:  # Process top 3 candidates
            plate_region = candidate['region']
            bbox = candidate['bbox']
            
            # Extract text from plate
            plate_text = self.extract_text_from_plate(plate_region)
            
            if plate_text:
                # Validate Nigerian plate format
                is_valid, confidence = self.validate_nigerian_plate(plate_text)
                
                if is_valid:
                    # Save detection
                    self.save_detection(image, plate_text, bbox, confidence)
                    
                    detections.append({
                        'plate_number': plate_text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'timestamp': datetime.now()
                    })
                    
        return detections
        
    def run_continuous_detection(self):
        """Run continuous plate detection"""
        self.logger.info("Starting continuous plate detection...")
        
        try:
            while True:
                # Capture frame
                frame = self.capture_frame()
                
                if frame is not None:
                    # Process frame
                    detections = self.process_frame(frame)
                    
                    # Display results (optional - comment out for headless operation)
                    if detections:
                        display_frame = frame.copy()
                        for detection in detections:
                            x, y, w, h = detection['bbox']
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_frame, 
                                      f"{detection['plate_number']} ({detection['confidence']:.2f})", 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Uncomment to show display (requires desktop environment)
                        # cv2.imshow('Plate Detection', cv2.resize(display_frame, (800, 600)))
                        
                    # Break on 'q' key (only works with display)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                        
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
        finally:
            self.cleanup()
            
    def capture_single_detection(self):
        """Capture and process a single frame"""
        frame = self.capture_frame()
        if frame is not None:
            detections = self.process_frame(frame)
            return detections
        return []
        
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera_type == "picamera":
                self.camera.stop()
            else:
                self.camera.release()
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    """Main function to run the plate recognition system"""
    try:
        # Initialize the recognition system
        recognizer = NigerianPlateRecognizer()
        
        # Choose operation mode
        print("Nigerian Vehicle Plate Recognition System")
        print("1. Continuous Detection")
        print("2. Single Shot Detection")
        choice = input("Select mode (1 or 2): ").strip()
        
        if choice == "1":
            recognizer.run_continuous_detection()
        elif choice == "2":
            detections = recognizer.capture_single_detection()
            if detections:
                print(f"Detected plates:")
                for detection in detections:
                    print(f"- {detection['plate_number']} (Confidence: {detection['confidence']:.2f})")
            else:
                print("No plates detected")
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"System error: {e}")
        
if __name__ == "__main__":
    main()