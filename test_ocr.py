"""
Test OCR Engine
Quick test script for OCR accuracy
"""

import cv2
import sys
import argparse
from pathlib import Path
from ocr_engine import OCREngine

def test_ocr(image_path, languages=['en'], gpu=True):
    """Test OCR on a single image"""
    print(f"Testing OCR on: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Initialize OCR
    print("Initializing OCR engine...")
    ocr = OCREngine(languages=languages, gpu=gpu)
    
    # Run OCR
    print("Running OCR...")
    result = ocr.read_license_plate(image, preprocessing=True)
    
    # Display results
    print("\n" + "="*50)
    print("OCR Results:")
    print("="*50)
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Valid: {result['valid']}")
    print("="*50)
    
    # Display image with result
    cv2.putText(
        image,
        f"Text: {result['text']} ({result['confidence']:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    cv2.imshow("OCR Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_directory(directory, languages=['en'], gpu=True):
    """Test OCR on all images in directory"""
    print(f"Testing OCR on directory: {directory}")
    
    # Initialize OCR
    ocr = OCREngine(languages=languages, gpu=gpu)
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(directory).glob(ext))
    
    if not image_files:
        print("No images found in directory")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    results = []
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        result = ocr.read_license_plate(image, preprocessing=True)
        results.append({
            'file': img_path.name,
            'text': result['text'],
            'confidence': result['confidence'],
            'valid': result['valid']
        })
        
        print(f"{img_path.name}: {result['text']} ({result['confidence']:.2f})")
    
    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    valid_count = sum(1 for r in results if r['valid'])
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    print(f"Total images: {len(results)}")
    print(f"Valid plates: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.2f}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Test OCR Engine')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--languages', nargs='+', default=['en'], help='OCR languages')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    
    args = parser.parse_args()
    
    if args.image:
        test_ocr(args.image, languages=args.languages, gpu=not args.no_gpu)
    elif args.directory:
        test_directory(args.directory, languages=args.languages, gpu=not args.no_gpu)
    else:
        print("Please specify --image or --directory")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
