import string
import easyocr
import cv2
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Enhanced Nigerian license plate character mapping for OCR corrections
dict_char_to_int = {
    'O': '0', 'o': '0',
    'I': '1', 'l': '1', 'i': '1',
    'J': '3', 'j': '3',
    'A': '4', 'a': '4',
    'G': '6', 'g': '6',
    'S': '5', 's': '5',
    'Z': '2', 'z': '2',
    'B': '8', 'b': '8',
    'D': '0', 'd': '0',  # Common OCR mistake
    'Q': '0', 'q': '0',  # Common OCR mistake
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '2': 'Z',
    '8': 'B'
}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )


def license_complies_nigeria_format(text):
    """
    ENHANCED: Check if the license plate text complies with Nigerian license plate format.
    More flexible patterns to handle OCR errors and variations.
    """
    if not text or len(text) < 6 or len(text) > 10:
        print(f"❌ Text length invalid: {len(text) if text else 0}")
        return False
    
    # Remove spaces, hyphens, and dots
    text = text.replace(' ', '').replace('-', '').replace('.', '').replace(',', '')
    
    # Enhanced Nigerian patterns (more flexible)
    patterns = [
        # Standard formats
        r'^[A-Z]{3}[0-9]{3}[A-Z]{2}$',     # ABC123XY (8 chars - most common)
        r'^[A-Z]{2}[0-9]{3}[A-Z]{3}$',     # AB123CDE (8 chars - new format)
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',        # ABC123X (7 chars)
        r'^[A-Z]{3}[0-9]{2}[A-Z]{2}$',     # ABC12XY (7 chars)
        
        # Commercial/government variants
        r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]$', # AB12CD3 (commercial)
        r'^[A-Z]{4}[0-9]{3}$',              # ABCD123 (some states)
        r'^[A-Z]{2}[0-9]{4}[A-Z]$',         # AB1234C
        r'^[A-Z]{3}[0-9]{4}$',              # ABC1234 (older format)
        
        # More flexible patterns to catch OCR errors
        r'^[A-Z]{2,4}[0-9]{2,4}[A-Z]{1,3}$', # Flexible: 2-4 letters, 2-4 numbers, 1-3 letters
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            print(f"✅ Text '{text}' matches Nigerian pattern: {pattern}")
            return True
    
    # If no exact match, check if it's "close enough" (for OCR errors)
    if is_likely_nigerian_plate(text):
        print(f"✅ Text '{text}' is likely a Nigerian plate (fuzzy match)")
        return True
    
    print(f"❌ Text '{text}' doesn't match any Nigerian license plate pattern")
    return False


def is_likely_nigerian_plate(text):
    """
    Check if text is likely a Nigerian license plate even with OCR errors
    """
    if not text or len(text) < 6 or len(text) > 10:
        return False
    
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    
    # Nigerian plates typically have 3-5 letters and 2-4 digits
    if letter_count >= 3 and digit_count >= 2:
        # Check if it has a reasonable structure
        has_letters_start = text[0].isalpha()
        has_digits_middle = any(c.isdigit() for c in text[2:6])
        
        if has_letters_start and has_digits_middle:
            return True
    
    return False


def format_nigeria_license(text):
    """
    Enhanced formatting for Nigerian license plates with better OCR correction
    """
    if not text:
        return text
        
    # Clean the text first
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(',', '')
    
    # Apply OCR corrections
    corrected_text = ''
    for char in text:
        corrected_text += char
    
    return corrected_text


def smart_ocr_correction(text):
    """
    Apply intelligent OCR corrections based on position and context
    """
    if not text or len(text) < 6:
        return text
        
    corrected = list(text.upper())
    
    # For 8-character plates (ABC123XY format)
    if len(corrected) == 8:
        # First 3 positions should be letters
        for i in range(3):
            if corrected[i].isdigit() and corrected[i] in dict_int_to_char:
                corrected[i] = dict_int_to_char[corrected[i]]
                print(f"🔧 Position {i}: '{text[i]}' -> '{corrected[i]}'")
        
        # Middle 3 positions (3,4,5) should be digits
        for i in range(3, 6):
            if corrected[i].isalpha() and corrected[i] in dict_char_to_int:
                corrected[i] = dict_char_to_int[corrected[i]]
                print(f"🔧 Position {i}: '{text[i]}' -> '{corrected[i]}'")
        
        # Last 2 positions should be letters
        for i in range(6, 8):
            if corrected[i].isdigit() and corrected[i] in dict_int_to_char:
                corrected[i] = dict_int_to_char[corrected[i]]
                print(f"🔧 Position {i}: '{text[i]}' -> '{corrected[i]}'")
    
    # For 7-character plates (ABC123X format)
    elif len(corrected) == 7:
        # First 3 should be letters
        for i in range(3):
            if corrected[i].isdigit() and corrected[i] in dict_int_to_char:
                corrected[i] = dict_int_to_char[corrected[i]]
                print(f"🔧 Position {i}: '{text[i]}' -> '{corrected[i]}'")
        
        # Middle 3 should be digits
        for i in range(3, 6):
            if corrected[i].isalpha() and corrected[i] in dict_char_to_int:
                corrected[i] = dict_char_to_int[corrected[i]]
                print(f"🔧 Position {i}: '{text[i]}' -> '{corrected[i]}'")
        
        # Last 1 should be letter
        if corrected[6].isdigit() and corrected[6] in dict_int_to_char:
            corrected[6] = dict_int_to_char[corrected[6]]
            print(f"🔧 Position 6: '{text[6]}' -> '{corrected[6]}'")
    
    return ''.join(corrected)


def preprocess_license_plate_image(image):
    """
    Enhanced image preprocessing for better OCR accuracy
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Multiple preprocessing approaches
    processed_images = []
    
    # 1. Basic grayscale
    processed_images.append(("grayscale", gray))
    
    # 2. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
   # processed_images.append(("adaptive_threshold", adaptive_thresh))
    
    # 3. OTSU thresholding
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #processed_images.append(("otsu_threshold", otsu_thresh))
    
    # 4. Histogram equalization
    equalized = cv2.equalizeHist(gray)
   # processed_images.append(("equalized", equalized))
    
    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    processed_images.append(("clahe", clahe_img))
    
    # 6. Gaussian blur + threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   # processed_images.append(("blur_threshold", blur_thresh))
    
    return processed_images


def read_license_plate(license_plate_crop):
    """
    ENHANCED: Read Nigerian license plate with multiple preprocessing and correction strategies
    """
    
    print(f"🇳🇬 Reading Nigerian license plate...")
    print(f"🔍 Input image shape: {license_plate_crop.shape}")
    
    # Resize image if too small (EasyOCR works better with larger images)
    height, width = license_plate_crop.shape[:2]
    min_height, min_width = 80, 200
    
    if height < min_height or width < min_width:
        scale_factor = max(min_height/height, min_width/width, 2.5)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        license_plate_crop = cv2.resize(license_plate_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"🔍 Resized image to: {license_plate_crop.shape}")
    
    # Get multiple preprocessed versions
    processed_images = preprocess_license_plate_image(license_plate_crop)
    
    # Store ALL potential license plates (valid and invalid) with scores
    all_detections = []
    
    try:
        for method_name, img in processed_images:
            print(f"🔬 Trying OCR on {method_name} image...")
            
            # Multiple OCR configurations
            ocr_configs = [
                {
                    'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    'width_ths': 0.7,
                    'height_ths': 0.7
                },
                {
                    'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    'width_ths': 0.5,
                    'height_ths': 0.5
                },
                {
                    'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    'width_ths': 0.9,
                    'height_ths': 0.9
                }
            ]
            
            for config_idx, config in enumerate(ocr_configs):
                try:
                    detections = reader.readtext(img, **config)
                    
                    print(f"🔍 Config {config_idx+1}: Found {len(detections)} detections in {method_name}")
                    
                    for i, detection in enumerate(detections):
                        bbox, text, score = detection
                        print(f"🔍 Detection {i+1}: '{text}' (confidence: {score:.3f})")
                        
                        # Clean up the text
                        cleaned_text = text.upper().replace(' ', '').replace('-', '').replace('.', '').replace(',', '')
                        
                        if len(cleaned_text) >= 6 and len(cleaned_text) <= 10:
                            # Apply smart OCR correction
                            corrected_text = smart_ocr_correction(cleaned_text)
                            print(f"🔍 Corrected text: '{corrected_text}'")
                            
                            # Store all detections with their info
                            all_detections.append({
                                'text': corrected_text,
                                'score': score,
                                'method': method_name,
                                'config': config_idx,
                                'is_valid': license_complies_nigeria_format(corrected_text)
                            })
                            
                except Exception as e:
                    print(f"❌ OCR config {config_idx+1} failed: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ EasyOCR error: {e}")
    
    if not all_detections:
        print("❌ NO license plate text detected at all!")
        return None, 0.0
    
    # Separate valid and invalid detections
    valid_detections = [d for d in all_detections if d['is_valid']]
    
    print(f"📊 Detection Summary:")
    print(f"   Total detections: {len(all_detections)}")
    print(f"   Valid Nigerian plates: {len(valid_detections)}")
    
    # If we have valid detections, return the best one
    if valid_detections:
        # Sort by confidence score
        best_valid = max(valid_detections, key=lambda x: x['score'])
        
        print(f"🏆 BEST VALID RESULT: '{best_valid['text']}' (score: {best_valid['score']:.3f})")
        return best_valid['text'], best_valid['score']
    
    # If no valid detections, return the best detection anyway (with lower confidence)
    else:
        best_detection = max(all_detections, key=lambda x: x['score'])
        print(f"⚠️ NO VALID plates found. Best guess: '{best_detection['text']}' (score: {best_detection['score']:.3f})")
        print(f"⚠️ This detection doesn't match Nigerian format but might be OCR error")
        
        # Return it with reduced confidence
        return best_detection['text'], best_detection['score'] * 0.5


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1