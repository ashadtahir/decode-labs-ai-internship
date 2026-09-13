# Project 4: Image / Text Recognition (OCR)

DecodeLabs AI Internship

---

## Project Title

Image / Text Recognition Using OCR (Optical Character Recognition)

---

## Objective

Build a simple application that reads (recognizes) the text inside an image using a pre-trained OCR engine. The application loads a sample image, pre-processes it to make the text clearer, runs OCR to extract the text, and validates that the OCR result meets an 80% minimum confidence requirement.

---

## Problem

Computers store images as grids of colored pixels. A computer cannot "read" text in an image by itself — it just sees numbers. We need a way to convert the text shown in an image into actual, editable text that a program can use.

---

## Solution

We use OCR (Optical Character Recognition) through a tool called **Tesseract**. Tesseract is a pre-trained OCR engine that has already learned how text looks, so no machine learning training is needed. Our program wraps Tesseract with **pytesseract** (the Python interface) and uses **OpenCV** to prepare the image so Tesseract gets the best possible input.

---

## What OCR Is

OCR (Optical Character Recognition) is the process of turning images of text into machine-readable text. "Optical" means visual/light-based, "Character" means letters and numbers, and "Recognition" means identifying them. In simple terms: OCR looks at a picture of words and outputs the words as text.

---

## Technologies Used

- **Python** — the programming language
- **OpenCV (cv2)** — image processing (grayscale, blur, deskew, threshold)
- **pytesseract** — Python wrapper that calls the Tesseract OCR engine
- **Tesseract OCR** — the actual pre-trained OCR engine (needs a separate install on Windows)
- **Pillow** — used to create the sample image with clear text

There is no model training, no neural networks, no databases, and no web/GUI layer.

---

## OCR Execution Path

This project follows **Path 1 - OCR** from the project brief. The simpler OCR path was chosen because it fully satisfies the requirements. The alternative object-detection path (MobileNet-SSD) is not used.

---

## Complete Processing Pipeline

```
Input Image
    |
    v
Load Image
    |
    v
Grayscale Conversion
    |
    v
Gaussian Blur
    |
    v
Deskewing
    |
    v
Thresholding
    |
    v
OCR using pytesseract
    |
    v
Confidence Validation
    |
    v
Display Final Text
```

All of these steps are actually implemented in `recognition.py`, not just described.

---

## Image Pre-Processing Steps

### 1. Grayscale Conversion

Color images have 3 channels (Blue, Green, Red). Grayscale conversion reduces the image to a single channel that only represents brightness. This makes the image simpler and faster for OCR to process, because color is not needed to read text.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

### 2. Gaussian Blur

Gaussian blur smooths the image by averaging nearby pixels. This removes small noise and rough edges so the text outlines become cleaner, which helps OCR separate letters from the background.

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```

### 3. Deskewing

Sometimes a scanned image is slightly rotated, which makes OCR less accurate. The deskew step detects the angle of the text using the smallest rotated rectangle that fits around the text pixels (`cv2.minAreaRect`) and rotates the image back so the lines of text are horizontal.

```python
coords = np.column_stack(np.where(image < 128))
angle = cv2.minAreaRect(coords)[-1]
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, matrix, (w, h), ...)
```

### 4. Thresholding

Thresholding turns the grayscale image into a strict black-and-white image. Adaptive thresholding compares each pixel with the average of its neighborhood and sets it to black or white based on a threshold. This strongly separates the dark text from the light background, which is exactly what OCR likes best.

```python
thresh = cv2.adaptiveThreshold(
    deskewed, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    35, 10,
)
```

---

## How pytesseract Is Used

`pytesseract` is a thin Python wrapper around the Tesseract command-line tool. The program runs OCR with:

```python
data = pytesseract.image_to_data(
    thresh,
    config="--psm 6",
    output_type=pytesseract.Output.DICT,
)
```

It uses **PSM (Page Segmentation Mode) 6**, which tells Tesseract to assume the image is a single uniform block of text — exactly right for the simple sample image. `image_to_data` returns details for every detected word, including each word's confidence score.

The recognized text lines are reconstructed from the returned data (grouped by block, paragraph, and line) so the original line layout is preserved. The confidence values come directly from Tesseract — nothing is hard-coded.

---

## Confidence Validation

For every recognized word, Tesseract returns a confidence score from 0 to 100 (higher is better). The program:

1. Collects the confidence of every word Tesseract actually detected (ignoring `-1` entries where nothing was recognized)
2. Computes the **average word confidence**
3. Compares that average against the required **80% minimum**

```python
all_conf = [conf for line_words in lines.values() for _, conf in line_words]
avg_confidence = sum(all_conf) / len(all_conf)
```

If the result is 80% or higher, the validation is marked **PASS**; otherwise it is marked **FAIL**.

---

## 80% Confidence Requirement

The DecodeLabs brief requires the OCR result to be validated against a minimum confidence of 80%. The program uses the **real** confidence scores returned by Tesseract and prints whether the requirement is met:

```
OCR Confidence: 95.50%
Validation: PASS - confidence is 80% or higher
```

---

## Sample Input

The project includes `sample.png`, a high-contrast image (black text on a white background, slightly tilted) containing:

```
DecodeLabs AI Internship
Project 4
Image Text Recognition
```

The slight tilt intentionally demonstrates the deskewing step.

---

## Installation

```bash
pip install -r requirements.txt
```

This installs:

```
opencv-python
pytesseract
Pillow
```

---

## Important: Tesseract OCR Setup on Windows

`pytesseract` is only a wrapper — the **Tesseract engine itself** must also be installed on Windows:

1. Download the official Windows installer:
   https://github.com/UB-Mannheim/tesseract/releases
   (choose the `tesseract-ocr-w64-setup-...` installer)
2. Run the installer and note the install location (default: `C:\Program Files\Tesseract-OCR\`)
3. `recognition.py` automatically uses Tesseract from `C:\Program Files\Tesseract-OCR\tesseract.exe` when present
4. If Tesseract is missing, the program prints a clear message explaining that it must be installed — it does not crash silently

If Tesseract is installed somewhere else, update `TESSERACT_PATH` in `recognition.py` to point to your `tesseract.exe`.

---

## How to Run

Make sure `sample.png` is in the same folder as `recognition.py`, then run:

```bash
cd Project-4-Image-Text-Recognition
python recognition.py
```

---

## Example Output

This is the real output produced by running `python recognition.py`:

```
============================================================
      IMAGE / TEXT RECOGNITION (OCR)
============================================================

[1] Input image loaded: sample.png
    Image size: 1400 x 400, channels: 3

[2] Converted to grayscale.

[3] Applied Gaussian blur to reduce noise.

[4] Deskewed image (corrected tilt of -2.12 degrees).

[5] Applied adaptive thresholding.

[6] Running OCR on the processed image (PSM 6)...

============================================================
                  FINAL RESULT
============================================================

Recognized Text:
------------------------------------------------------------
DecodeLabs AI Internship
Project 4
Image Text Recognition
------------------------------------------------------------

OCR Confidence: 95.50%
Validation: PASS - confidence is 80% or higher
============================================================
```

The four validation areas are demonstrated:

1. **Library integration** — pytesseract and OpenCV load and run successfully
2. **Pre-processing integrity** — grayscale, blur, deskew (real tilt corrected) and thresholding all execute
3. **Accuracy benchmarking** — real OCR confidence (95.50%) is validated against the 80% minimum
4. **Visual confirmation** — the recognized text is printed clearly on screen

---

## Project Structure

```
Project-4-Image-Text-Recognition/
├── recognition.py       # Main OCR application
├── sample.png           # Sample input image with readable text
├── requirements.txt     # Python packages to install
└── README.md            # This file
```

---

## Reflection

This project showed how a "computer vision" task can be solved with simple, proven tools rather than complex AI:

- **Pre-processing makes a real difference.** Converting to grayscale, blurring, deskewing, and thresholding gave Tesseract clean black-and-white text to work with — OCR quality depends heavily on input quality.
- **Deskewing works.** The sample image was rotated 2 degrees and the deskew step detected and corrected it before OCR.
- **Confidence is real feedback.** Reading actual per-word confidence from Tesseract makes the model's own belief transparent and lets us enforce the 80% requirement honestly.
- **OCR is not magic.** Despite clean input, the engine still needed hand-tuned parameters (blur kernel, threshold block size, PSM mode) to read perfectly — a great lesson in how preprocessing and configuration matter.
- **No training needed.** Using a pre-trained engine (Tesseract) for OCR was the right, simple path for this project.