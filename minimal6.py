#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import os
import threading
import queue
import time
from datetime import datetime
from utils3 import read_license_plate
from picamera2 import Picamera2
import numpy as np
import base64
from supabase import create_client, Client
import json
from typing import Optional, Tuple

class SupabaseManager:
    def __init__(self, supabase_url: str, supabase_key: str, bucket_name: str = "license-plates"):
        """
        Initialize Supabase client for database and storage operations
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase anon/service key
            bucket_name: Storage bucket name for images
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.bucket_name = bucket_name
        self.supabase: Optional[Client] = None
        
    def connect(self) -> bool:
        """Connect to Supabase"""
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection by trying to access storage
            buckets = self.supabase.storage.list_buckets()
            print(f"✅ Connected to Supabase successfully")
            print(f"📁 Available buckets: {[bucket.name for bucket in buckets]}")
            
            # Create bucket if it doesn't exist
            existing_buckets = [bucket.name for bucket in buckets]
            if self.bucket_name not in existing_buckets:
                try:
                    self.supabase.storage.create_bucket(self.bucket_name, {"public": True})
                    print(f"✅ Created bucket: {self.bucket_name}")
                except Exception as e:
                    print(f"⚠️ Bucket creation failed (may already exist): {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def upload_image(self, image_data: bytes, file_name: str) -> Optional[str]:
        """
        Upload image to Supabase storage
        
        Args:
            image_data: Image data as bytes
            file_name: Name for the file in storage
            
        Returns:
            Public URL of uploaded image or None if failed
        """
        try:
            # Upload file to storage
            result = self.supabase.storage.from_(self.bucket_name).upload(
                file_name, 
                image_data,
                file_options={"content-type": "image/jpeg"}
            )
            
            if result:
                # Get public URL
                public_url = self.supabase.storage.from_(self.bucket_name).get_public_url(file_name)
                print(f"✅ Image uploaded successfully: {file_name}")
                return public_url
            else:
                print(f"❌ Failed to upload image: {file_name}")
                return None
                
        except Exception as e:
            print(f"❌ Image upload error: {e}")
            return None
    
    def save_detection(self, license_plate_text: str, confidence_score: float, 
                      full_image_url: str, crop_image_url: str, location: dict = None) -> bool:
        """
        Save license plate detection to database
        
        Args:
            license_plate_text: Detected license plate text
            confidence_score: OCR confidence score
            full_image_url: URL of full detection image
            crop_image_url: URL of cropped license plate image
            location: Optional location data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            detection_data = {
                "license_plate_number": license_plate_text,
                "confidence_score": confidence_score,
                "full_image_url": full_image_url,
                "crop_image_url": crop_image_url,
                "detected_at": datetime.now().isoformat(),
                "location": location,
                "status": "active"
            }
            
            result = self.supabase.table("license_plate_detections").insert(detection_data).execute()
            
            if result.data:
                print(f"✅ Detection saved to database: {license_plate_text}")
                return True
            else:
                print(f"❌ Failed to save detection to database")
                return False
                
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return False


class LicensePlateDetector:
    def __init__(self, model_path='license_plate.pt', output_dir='detections', 
                 confidence_threshold=0.4, supabase_url=None, supabase_key=None):
        """
        Initialize the License Plate Detector for Raspberry Pi Camera with Supabase integration
        
        Args:
            model_path: Path to trained YOLO model
            output_dir: Directory to save detection results
            confidence_threshold: Minimum confidence for detections
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service key
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        
        # Supabase integration
        self.supabase_manager = None
        if supabase_url and supabase_key:
            self.supabase_manager = SupabaseManager(supabase_url, supabase_key)
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue()
        self.upload_queue = queue.Queue()  # Queue for Supabase uploads
        self.running = False
        
        # Camera setup
        self.picam2 = None
        self.model = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Detection tracking
        self.last_detection_time = 0
        self.detection_cooldown = 2.0
        
    def setup_camera(self, resolution=(640, 480)):
        """Initialize the Raspberry Pi camera"""
        try:
            self.picam2 = Picamera2()
            
            config = self.picam2.create_preview_configuration(
                main={"size": resolution, "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            print(f"✅ Pi Camera initialized with resolution: {resolution}")
            time.sleep(2)
            return True
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to("cpu")
            self.model.fuse()
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def setup_supabase(self):
        """Setup Supabase connection"""
        if self.supabase_manager:
            return self.supabase_manager.connect()
        return False
    
    def camera_capture_thread(self):
        """Thread for capturing frames from camera"""
        print("🎥 Camera capture thread started")
        
        while self.running:
            try:
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Show live camera feed
                if os.environ.get("DISPLAY", "") != "":
                    cv2.imshow("License plate feed", frame_bgr)
                    cv2.waitKey(1)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_bgr)
                else:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame_bgr)
                    except queue.Empty:
                        pass
                
                time.sleep(0.03)
                
            except Exception as e:
                print(f"❌ Camera capture error: {e}")
                break
        
        print("🎥 Camera capture thread stopped")
    
    def detection_thread(self):
        """Thread for processing license plate detection"""
        print("🔍 Detection thread started")
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_cooldown:
                    continue
                
                results = self.model(frame, device="cpu", half=True)
                
                detection_found = False
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf > self.confidence_threshold:
                                detection_found = True
                                self.process_detection(frame, box, conf)
                                self.last_detection_time = current_time
                                break
                    if detection_found:
                        break
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Detection error: {e}")
                continue
        
        print("🔍 Detection thread stopped")
    
    def supabase_upload_thread(self):
        """Thread for handling Supabase uploads"""
        print("☁️ Supabase upload thread started")
        
        while self.running:
            try:
                upload_data = self.upload_queue.get(timeout=1.0)
                self.handle_supabase_upload(upload_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Upload thread error: {e}")
        
        print("☁️ Supabase upload thread stopped")
    
    def handle_supabase_upload(self, upload_data):
        """Handle uploading data to Supabase"""
        try:
            license_plate_text = upload_data['license_plate_text']
            confidence_score = upload_data['confidence_score']
            full_image_path = upload_data['full_image_path']
            crop_image_path = upload_data['crop_image_path']
            timestamp = upload_data['timestamp']
            
            print(f"☁️ Uploading to Supabase: {license_plate_text}")
            
            # Read image files
            with open(full_image_path, 'rb') as f:
                full_image_data = f.read()
            
            with open(crop_image_path, 'rb') as f:
                crop_image_data = f.read()
            
            # Upload images to Supabase storage
            full_image_filename = f"full_detections/{timestamp}_full.jpg"
            crop_image_filename = f"plate_crops/{timestamp}_crop.jpg"
            
            full_image_url = self.supabase_manager.upload_image(full_image_data, full_image_filename)
            crop_image_url = self.supabase_manager.upload_image(crop_image_data, crop_image_filename)
            
            if full_image_url and crop_image_url:
                # Save detection data to database
                success = self.supabase_manager.save_detection(
                    license_plate_text=license_plate_text,
                    confidence_score=confidence_score,
                    full_image_url=full_image_url,
                    crop_image_url=crop_image_url,
                    location=None  # Add GPS coordinates if available
                )
                
                if success:
                    print(f"🎉 Successfully uploaded {license_plate_text} to Supabase!")
                else:
                    print(f"❌ Failed to save detection data for {license_plate_text}")
            else:
                print(f"❌ Failed to upload images for {license_plate_text}")
                
        except Exception as e:
            print(f"❌ Supabase upload error: {e}")
    
    def process_detection(self, frame, box, confidence):
        """Process a detected license plate"""
        try:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            print(f"🔍 License plate detected! Confidence: {confidence:.2f}")
            print(f"📍 Location: ({x1}, {y1}, {x2}, {y2})")
            
            if (x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and 
                x2 <= frame.shape[1] and y2 <= frame.shape[0]):
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Save full frame with detection
                full_frame_path = os.path.join(self.output_dir, f"detection_{timestamp}_full.jpg")
                frame_with_box = frame.copy()
                cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_with_box, f"Conf: {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(full_frame_path, frame_with_box)
                print(f"💾 Full detection saved: {full_frame_path}")
                
                # Crop license plate
                license_plate_crop = frame[y1:y2, x1:x2, :]
                
                if license_plate_crop.size > 0 and (x2-x1) > 20 and (y2-y1) > 10:
                    crop_path = os.path.join(self.output_dir, f"plate_crop_{timestamp}.jpg")
                    cv2.imwrite(crop_path, license_plate_crop)
                    print(f"💾 License plate crop saved: {crop_path}")
                    
                    # Process with OCR in separate thread
                    ocr_thread = threading.Thread(
                        target=self.process_ocr_with_supabase,
                        args=(license_plate_crop, timestamp, full_frame_path, crop_path),
                        daemon=True
                    )
                    ocr_thread.start()
                    
                else:
                    print("⚠️ License plate crop too small")
            else:
                print("⚠️ Invalid crop coordinates")
                
        except Exception as e:
            print(f"❌ Error processing detection: {e}")
    
    def process_ocr_with_supabase(self, license_plate_crop, timestamp, full_frame_path, crop_path):
        """Process OCR and handle Supabase upload if Nigerian format is matched"""
        try:
            print(f"🔤 Processing OCR for detection {timestamp}...")
            
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
            
            print(f"📋 OCR Result for {timestamp}:")
            if license_plate_text:
                print(f"   Text: '{license_plate_text}'")
                print(f"   Score: {license_plate_text_score:.3f}")
                
                # Import the function from utils3
                from utils3 import license_complies_nigeria_format
                
                # Check if it matches Nigerian format
                is_nigerian_format = license_complies_nigeria_format(license_plate_text)
                
                if is_nigerian_format and self.supabase_manager:
                    print(f"🇳🇬 Nigerian license plate detected! Preparing for Supabase upload...")
                    
                    # Add to upload queue for Supabase
                    upload_data = {
                        'license_plate_text': license_plate_text,
                        'confidence_score': license_plate_text_score,
                        'full_image_path': full_frame_path,
                        'crop_image_path': crop_path,
                        'timestamp': timestamp
                    }
                    
                    self.upload_queue.put(upload_data)
                    print(f"📤 Added to Supabase upload queue: {license_plate_text}")
                    
                elif is_nigerian_format:
                    print(f"🇳🇬 Nigerian format detected but Supabase not configured")
                else:
                    print(f"❌ Not a valid Nigerian license plate format - skipping Supabase upload")
                
                # Save OCR results locally regardless
                results_file = os.path.join(self.output_dir, f"ocr_results_{timestamp}.txt")
                with open(results_file, 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"License Plate Text: {license_plate_text}\n")
                    f.write(f"Confidence Score: {license_plate_text_score:.3f}\n")
                    f.write(f"Nigerian Format: {is_nigerian_format}\n")
                    f.write(f"Supabase Upload: {'Queued' if is_nigerian_format and self.supabase_manager else 'Skipped'}\n")
                
                print(f"💾 OCR results saved: {results_file}")
            else:
                print(f"   Text: No valid license plate detected")
                print(f"   Score: {license_plate_text_score:.3f}")
                
                results_file = os.path.join(self.output_dir, f"ocr_failed_{timestamp}.txt")
                with open(results_file, 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"License Plate Text: DETECTION_FAILED\n")
                    f.write(f"Confidence Score: {license_plate_text_score:.3f}\n")
                
                print(f"💾 Failed OCR attempt logged: {results_file}")
            
        except Exception as e:
            print(f"❌ OCR processing error: {e}")
    
    def start_detection(self):
        """Start the license plate detection system"""
        print("🚀 Starting License Plate Detection System with Supabase Integration")
        print("=" * 60)
        
        # Initialize camera
        if not self.setup_camera():
            return False
        
        # Load model
        if not self.load_model():
            return False
        
        # Setup Supabase if configured
        if self.supabase_manager:
            if self.setup_supabase():
                print("✅ Supabase connected - Nigerian plates will be uploaded")
            else:
                print("❌ Supabase connection failed - running in local mode only")
                self.supabase_manager = None
        else:
            print("⚠️ Supabase not configured - running in local mode only")
        
        self.running = True
        
        # Start threads
        camera_thread = threading.Thread(target=self.camera_capture_thread, daemon=True)
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        
        camera_thread.start()
        detection_thread.start()
        
        # Start Supabase upload thread if configured
        if self.supabase_manager:
            upload_thread = threading.Thread(target=self.supabase_upload_thread, daemon=True)
            upload_thread.start()
        
        print("✅ All threads started successfully!")
        print("📷 Camera is monitoring for Nigerian license plates...")
        print("☁️ Valid detections will be uploaded to Supabase")
        print("⌨️ Press Ctrl+C to quit")
        
        try:
            while self.running:
                time.sleep(0.5)
                    
        except KeyboardInterrupt:
            print("\n⚡ Keyboard interrupt received")
        
        self.stop_detection()
        return True
    
    def show_status(self):
        """Show current system status"""
        print("\n" + "=" * 40)
        print("📊 System Status:")
        print(f"🎥 Camera: {'Running' if self.running else 'Stopped'}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"⏱️ Detection Cooldown: {self.detection_cooldown}s")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}")
        print(f"☁️ Supabase: {'Connected' if self.supabase_manager else 'Not configured'}")
        frame_queue_size = self.frame_queue.qsize()
        upload_queue_size = self.upload_queue.qsize() if hasattr(self, 'upload_queue') else 0
        print(f"📦 Frame Queue: {frame_queue_size}")
        print(f"📤 Upload Queue: {upload_queue_size}")
        print("=" * 40 + "\n")
    
    def stop_detection(self):
        """Stop the detection system"""
        print("\n🛑 Stopping detection system...")
        self.running = False
        
        time.sleep(2)  # Wait for threads to finish
        
        if self.picam2:
            try:
                self.picam2.stop()
                print("📷 Camera stopped")
            except:
                pass
        
        print("✅ Detection system stopped successfully")


def main():
    """Main function"""
    # Configuration
    MODEL_PATH = "license_plate.pt"
    OUTPUT_DIR = "live_detections"
    CONFIDENCE_THRESHOLD = 0.4
    
    # Supabase configuration - Replace with your actual credentials
    SUPABASE_URL = "https://vzlopjwgzwlcetmbkfzg.supabase.co"  # Replace with your Supabase URL
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ6bG9wandnendsY2V0bWJrZnpnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2NzU1MzMsImV4cCI6MjA3MzI1MTUzM30.Y0MK2sayrgQYEmWRdXkmMk-3n01ix4kC0PfWyWxXGeA"  # Replace with your Supabase anon key
    
    # Create detector instance with Supabase integration
    detector = LicensePlateDetector(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY
    )
    
    # Start detection
    try:
        detector.start_detection()
    except Exception as e:
        print(f"❌ Error starting detection: {e}")
    finally:
        detector.stop_detection()


if __name__ == "__main__":
    main()