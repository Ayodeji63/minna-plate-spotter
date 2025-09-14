from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import time
import threading
import queue

import util
from sort.sort import *
from util import get_car, read_license_plate

class LicensePlateDetector:
    def __init__(self):
        self.results = {}
        self.mot_tracker = Sort()
        self.detected_plates = set()  # To avoid duplicate logging
        self.plate_queue = queue.Queue()
        
        # Load models
        print("Loading models...")
        self.coco_model = YOLO('yolo11n.pt')
        self.license_plate_detector = YOLO('license_plate.pt')
        
        # Vehicle class IDs (car, motorcycle, bus, truck)
        self.vehicles = [2, 3, 5, 7]
        
        # Start file writer thread
        self.writer_thread = threading.Thread(target=self.write_plates_to_file, daemon=True)
        self.writer_thread.start()
        
        print("Models loaded successfully!")

    def write_plates_to_file(self):
        """Background thread to write detected plates to file"""
        with open('detected_plates.txt', 'a') as f:
            while True:
                try:
                    plate_data = self.plate_queue.get(timeout=1)
                    if plate_data is None:  # Shutdown signal
                        break
                    f.write(f"{plate_data}\n")
                    f.flush()
                    print(f"Logged: {plate_data}")
                except queue.Empty:
                    continue

    def log_license_plate(self, plate_text, confidence):
        """Log detected license plate with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plate_entry = f"{timestamp} - Plate: {plate_text} (Confidence: {confidence:.2f})"
        
        # Only log unique plates (avoid spam from same plate in consecutive frames)
        plate_key = f"{plate_text}_{int(time.time() // 5)}"  # 5-second window
        if plate_key not in self.detected_plates:
            self.detected_plates.add(plate_key)
            self.plate_queue.put(plate_entry)

    def test_camera_access(self):
        """Test different camera access methods"""
        print("Testing camera access methods...")
        
        # Test different camera indices and backends
        camera_configs = [
            (0, cv2.CAP_V4L2),      # V4L2 backend (most common on Linux)
            (0, cv2.CAP_GSTREAMER), # GStreamer backend
            (0, cv2.CAP_ANY),       # Any available backend
            ('/dev/video0', cv2.CAP_V4L2), # Direct device path
        ]
        
        for i, (camera_id, backend) in enumerate(camera_configs):
            try:
                print(f"Trying method {i+1}: {camera_id} with backend {backend}")
                cap = cv2.VideoCapture(camera_id, backend)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Success with method {i+1}")
                        cap.release()
                        return camera_id, backend
                    else:
                        print(f"✗ Camera opened but no frame captured")
                else:
                    print(f"✗ Could not open camera")
                cap.release()
            except Exception as e:
                print(f"✗ Error with method {i+1}: {e}")
        
        return None, None

    def run_detection(self):
        # Initialize Pi Camera
        try:
            # First, try legacy PiCamera
            camera_method = None
            try:
                from picamera import PiCamera
                from picamera.array import PiRGBArray
                
                print("Attempting to use legacy PiCamera...")
                camera = PiCamera()
                camera.resolution = (640, 480)
                camera.framerate = 30
                rawCapture = PiRGBArray(camera, size=(640, 480))
                
                print("✓ Using legacy PiCamera successfully!")
                time.sleep(0.1)  # Camera warm-up
                
                frame_count = 0
                process_every_n_frames = 2
                
                try:
                    for frame_capture in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                        frame = frame_capture.array
                        frame_count += 1
                        
                        if frame_count % process_every_n_frames == 0:
                            self.process_frame(frame, frame_count)
                        
                        cv2.imshow('License Plate Detection', frame)
                        rawCapture.truncate(0)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                except KeyboardInterrupt:
                    print("Detection stopped by user")
                finally:
                    camera.close()
                    
                camera_method = "legacy_picamera"
                        
            except ImportError:
                print("Legacy PiCamera not available, trying OpenCV methods...")
                
                # Test camera access
                camera_id, backend = self.test_camera_access()
                
                if camera_id is None:
                    print("\n" + "="*50)
                    print("CAMERA TROUBLESHOOTING:")
                    print("="*50)
                    print("1. Check if camera is connected:")
                    print("   ls /dev/video*")
                    print("")
                    print("2. Check camera status:")
                    print("   vcgencmd get_camera")
                    print("")
                    print("3. Enable camera interface:")
                    print("   sudo raspi-config")
                    print("   Interface Options -> Camera -> Enable")
                    print("")
                    print("4. For libcamera (newer Pi OS):")
                    print("   libcamera-hello --list-cameras")
                    print("")
                    print("5. Reboot after enabling camera:")
                    print("   sudo reboot")
                    print("="*50)
                    return
                
                print(f"✓ Using OpenCV with camera {camera_id} and backend {backend}")
                cap = cv2.VideoCapture(camera_id, backend)
                
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
                
                # Verify settings
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera settings: {int(actual_width)}x{int(actual_height)} @ {actual_fps}fps")
                
                frame_count = 0
                process_every_n_frames = 2
                consecutive_failures = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_failures += 1
                            print(f"Frame capture failed ({consecutive_failures}/5)")
                            
                            if consecutive_failures >= 5:
                                print("Too many consecutive failures. Stopping...")
                                break
                            continue
                        
                        consecutive_failures = 0  # Reset failure counter
                        frame_count += 1
                        
                        if frame_count % process_every_n_frames == 0:
                            self.process_frame(frame, frame_count)
                        
                        cv2.imshow('License Plate Detection', frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                except KeyboardInterrupt:
                    print("Detection stopped by user")
                finally:
                    cap.release()
                    
                camera_method = "opencv"
                
        except Exception as e:
            print(f"Camera initialization error: {e}")
            print("\nTrying alternative camera detection...")
            
            # Last resort: try all video devices
            import glob
            video_devices = glob.glob('/dev/video*')
            print(f"Available video devices: {video_devices}")
            
            for device in video_devices:
                try:
                    print(f"Trying device: {device}")
                    cap = cv2.VideoCapture(device)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            print(f"✓ {device} works!")
                            cap.release()
                            break
                    cap.release()
                except:
                    continue
            
        finally:
            cv2.destroyAllWindows()
            # Signal writer thread to shutdown
            self.plate_queue.put(None)
            
            if hasattr(self, 'writer_thread'):
                self.writer_thread.join(timeout=2)

    def process_frame(self, frame, frame_nmr):
        """Process a single frame for license plate detection"""
        try:
            # Detect vehicles
            detections = self.coco_model(frame, verbose=False)[0]
            detections_ = []
            
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in self.vehicles and score > 0.5:  # Confidence threshold
                    detections_.append([x1, y1, x2, y2, score])

            if len(detections_) == 0:
                return

            # Track vehicles
            track_ids = self.mot_tracker.update(np.asarray(detections_))

            # Detect license plates
            license_plates = self.license_plate_detector(frame, verbose=False)[0]
            
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                
                if score < 0.5:  # Skip low confidence detections
                    continue

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    
                    if license_plate_crop.size == 0:
                        continue

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                    )

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None and license_plate_text_score > 0.8:
                        # Log to file
                        self.log_license_plate(license_plate_text, license_plate_text_score)
                        
                        # Draw bounding box on frame (optional)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
        except Exception as e:
            print(f"Error processing frame {frame_nmr}: {e}")

def main():
    print("Starting Pi Camera License Plate Detection...")
    print("Press 'q' to quit")
    print("Detected plates will be saved to 'detected_plates.txt'")
    
    detector = LicensePlateDetector()
    detector.run_detection()
    
    print("Detection stopped. Check 'detected_plates.txt' for results.")

if __name__ == "__main__":
    main()