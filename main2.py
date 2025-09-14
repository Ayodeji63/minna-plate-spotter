#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import os
from util import read_license_plate

def detect_license_plate(image_path, model_path='license_plate.pt', output_dir='detections'):
    """
    Detect license plates in an image and save the result
    
    Args:
        image_path: Path to input image
        model_path: Path to trained YOLO model
        output_dir: Directory to save detection results
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return
    
    # Load your trained model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    frame = image.read()
    if image is None:
        print(f"❌ Error: Could not read image '{image_path}'")
        return
    
    print(f"📷 Processing image: {image_path}")
    print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run inference
    results = model(image)
    
    # Check for detections
    detection_count = 0
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf > 0.4:  # Confidence threshold
                    x1, y1, x2, y2, score, class_id = box
                    
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    
                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    
                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    
                    print(f"License plate text {license_plate_text}")
                    print(f"License plate score {license_plate_text_score}")
                    
                    detection_count += 1
                    print(f"🚗 License plate #{detection_count} detected! Confidence: {conf:.2f}")
    
    if detection_count == 0:
        print("❌ No license plates detected in the image")
        print("💡 Try adjusting the confidence threshold or check if the image contains license plates")
    else:
        print(f"🎯 Total detections: {detection_count}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get annotated image with bounding boxes
    annotated_image = results[0].plot()
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    
    # Save the result
    cv2.imwrite(output_path, annotated_image)
    print(f"💾 Detection result saved to: {output_path}")
    
    # Display the result (optional - comment out if running headless)
    cv2.imshow('Original Image', image)
    cv2.imshow('License Plate Detection', annotated_image)
    print("👁️  Press any key to close the image windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return output_path

# Main execution
if __name__ == "__main__":
    # Example usage - modify these paths
    IMAGE_PATH = "6.JPG"  # Change to your image path
    MODEL_PATH = "license_plate.pt"         # Your trained model
    
    print("🚀 Starting License Plate Detection from Image")
    print("=" * 50)
    
    # You can also specify the image path as command line argument
    import sys
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    
    # Run detection
    result = detect_license_plate(IMAGE_PATH, MODEL_PATH)
    
    if result:
        print("=" * 50)
        print("✅ Detection completed successfully!")
        print(f"📁 Check the 'detections' folder for results")
    else:
        print("❌ Detection failed!")

