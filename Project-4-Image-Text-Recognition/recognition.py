# Image / Text Recognition - Project 4
# DecodeLabs AI Internship - OCR execution path

import os
import cv2
import numpy as np
import pytesseract

# Point pytesseract to the Tesseract installation (Windows default location)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def deskew(image):
    """Correct slight rotations so the text lines up horizontally.

    Uses the angle of the smallest rotated rectangle around the text pixels.
    """
    # Find the coordinates of the dark (text) pixels
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 10:
        return image, 0.0

    # The smallest rectangle that can contain all text pixels
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize the angle to the [-45, 45] range
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only rotate if there is a meaningful tilt
    if abs(angle) < 0.5:
        return image, 0.0

    # Rotate by the detected angle to straighten the text
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def main():
    print("=" * 60)
    print("      IMAGE / TEXT RECOGNITION (OCR)")
    print("=" * 60)

    # Step 1: Load the input image
    img_path = "sample.png"
    if not os.path.exists(img_path):
        print(f"\n[ERROR] Image file '{img_path}' was not found.")
        print("        Make sure sample.png is in the same folder as recognition.py.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("\n[ERROR] Could not read the image. The file may be corrupted.")
        return

    print(f"\n[1] Input image loaded: {img_path}")
    print(f"    Image size: {img.shape[1]} x {img.shape[0]}, channels: {img.shape[2]}")

    # Step 2: Convert to grayscale - simplifies the image to one channel
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("\n[2] Converted to grayscale.")

    # Step 3: Gaussian blur - reduces noise and smooths the edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("\n[3] Applied Gaussian blur to reduce noise.")

    # Step 4: Deskew - correct any slight tilt in the text
    deskewed, angle = deskew(blurred)
    if angle != 0.0:
        print(f"\n[4] Deskewed image (corrected tilt of {angle:.2f} degrees).")
    else:
        print("\n[4] Deskew: no significant tilt detected, image left as is.")

    # Step 5: Adaptive thresholding - separates text from background
    thresh = cv2.adaptiveThreshold(
        deskewed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 10,
    )
    print("\n[5] Applied adaptive thresholding.")

    # Step 6: Run OCR - PSM 6 assumes a single uniform block of text
    print("\n[6] Running OCR on the processed image (PSM 6)...")
    try:
        data = pytesseract.image_to_data(
            thresh,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        print("\n[ERROR] Tesseract OCR was not found.")
        print("        Install Tesseract from https://github.com/UB-Mannheim/tesseract")
        print("        and make sure it is installed at:")
        print("        C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        return
    except pytesseract.TesseractError as e:
        print(f"\n[ERROR] Tesseract could not process the image: {e}")
        return

    # Step 7: Collect the recognized text and real confidence values
    # Group words by their detected line so line breaks are preserved
    lines = {}
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = float(data["conf"][i])
        # Tesseract returns -1 when there is no recognized element
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, [])
        lines[key].append((text.strip(), conf))

    if not lines:
        print("\n[ERROR] OCR produced no text. The image may be too hard to read.")
        return

    # Build the recognized text while keeping the original line layout
    recognized_lines = [" ".join(word for word, _ in line_words) for line_words in lines.values()]
    recognized_text = "\n".join(recognized_lines)

    # Average of all real word confidences returned by Tesseract
    all_conf = [conf for line_words in lines.values() for _, conf in line_words]
    avg_confidence = sum(all_conf) / len(all_conf)

    # Step 8: Validate against the required 80% confidence threshold
    passed = avg_confidence >= 80.0

    # Step 9: Display the final results
    print("\n" + "=" * 60)
    print("                  FINAL RESULT")
    print("=" * 60)
    print("\nRecognized Text:")
    print("-" * 60)
    print(recognized_text)
    print("-" * 60)
    print(f"\nOCR Confidence: {avg_confidence:.2f}%")
    if passed:
        print("Validation: PASS - confidence is 80% or higher")
    else:
        print("Validation: FAIL - confidence is below 80%")
    print("=" * 60)


if __name__ == "__main__":
    main()