"""
OCR Engine for License Plate Recognition
Uses EasyOCR with GPU acceleration and preprocessing for optimal accuracy
"""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
import easyocr

logger = logging.getLogger(__name__)


class OCREngine:
    """License plate OCR with EasyOCR"""
    
    def __init__(self, languages=['en'], gpu=True, confidence_threshold=0.7):
        """
        Initialize OCR engine
        
        Args:
            languages: List of language codes (e.g., ['en', 'hi'])
            gpu: Use GPU acceleration
            confidence_threshold: Minimum confidence for OCR results
        """
        self.languages = languages
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.reader = None
        
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader"""
        try:
            logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            logger.info(f"GPU enabled: {self.gpu}")
            
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            
            logger.info("✓ OCR engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            raise
    
    def read_text(self, image, preprocessing=True):
        """
        Extract text from image
        
        Args:
            image: Input image (numpy array)
            preprocessing: Apply preprocessing for better accuracy
            
        Returns:
            List of tuples: (text, confidence)
        """
        try:
            if image is None or image.size == 0:
                return []
            
            # Preprocess image if enabled
            if preprocessing:
                image = self._preprocess_image(image)
            
            # Run OCR
            results = self.reader.readtext(image)
            
            # Filter by confidence and format results
            filtered_results = []
            for (bbox, text, conf) in results:
                if conf >= self.confidence_threshold:
                    # Clean text (remove special characters, spaces)
                    cleaned_text = self._clean_text(text)
                    if cleaned_text:
                        filtered_results.append((cleaned_text, conf))
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []
    
    def read_license_plate(self, plate_image, preprocessing=True):
        """
        Extract license plate text with validation
        
        Args:
            plate_image: Cropped license plate image
            preprocessing: Apply preprocessing
            
        Returns:
            dict with keys:
                - text: Extracted text
                - confidence: OCR confidence
                - valid: Whether text matches license plate pattern
        """
        results = self.read_text(plate_image, preprocessing)
        
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'valid': False
            }
        
        # Get result with highest confidence
        best_result = max(results, key=lambda x: x[1])
        text, confidence = best_result
        
        # Validate license plate format
        is_valid = self._validate_license_plate(text)
        
        return {
            'text': text,
            'confidence': confidence,
            'valid': is_valid
        }
    
    def _preprocess_image(self, image):
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize to optimal size (width ~320px)
            height, width = gray.shape
            if width < 320:
                scale = 320 / width
                new_width = 320
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            return binary
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using original image")
            return image
    
    def _clean_text(self, text):
        """
        Clean OCR text output
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove spaces and special characters
        cleaned = ''.join(c for c in text if c.isalnum())
        
        # Convert to uppercase
        cleaned = cleaned.upper()
        
        return cleaned
    
    def _validate_license_plate(self, text):
        """
        Validate if text matches license plate pattern
        
        Args:
            text: Extracted text
            
        Returns:
            True if valid license plate format
        """
        # Basic validation: 
        # - Length between 4-12 characters
        # - Contains both letters and numbers
        
        if not text or len(text) < 4 or len(text) > 12:
            return False
        
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        return has_letter and has_digit
    
    def batch_read(self, images, preprocessing=True):
        """
        Process multiple images in batch
        
        Args:
            images: List of images
            preprocessing: Apply preprocessing
            
        Returns:
            List of results (one per image)
        """
        results = []
        
        for image in images:
            result = self.read_license_plate(image, preprocessing)
            results.append(result)
        
        return results


class PlatePreprocessor:
    """Advanced preprocessing for license plate images"""
    
    @staticmethod
    def enhance_plate(image):
        """
        Apply comprehensive enhancement to plate image
        
        Args:
            image: Input plate image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Perspective correction (if needed)
            corrected = PlatePreprocessor._correct_perspective(gray)
            
            # Super-resolution upscaling (simple bicubic)
            height, width = corrected.shape
            upscaled = cv2.resize(
                corrected,
                (width * 2, height * 2),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(upscaled)
            
            # Sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Denoising
            denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return image
    
    @staticmethod
    def _correct_perspective(image):
        """
        Attempt to correct perspective distortion
        
        Args:
            image: Input image
            
        Returns:
            Corrected image
        """
        try:
            # Find edges
            edges = cv2.Canny(image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return image
            
            # Find largest rectangular contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we found a quadrilateral, apply perspective transform
            if len(approx) == 4:
                # Order points: top-left, top-right, bottom-right, bottom-left
                pts = approx.reshape(4, 2)
                rect = PlatePreprocessor._order_points(pts)
                
                # Compute destination points
                width = 320
                height = 64
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Compute perspective transform
                M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
                warped = cv2.warpPerspective(image, M, (width, height))
                
                return warped
            
            return image
            
        except Exception as e:
            logger.debug(f"Perspective correction failed: {e}")
            return image
    
    @staticmethod
    def _order_points(pts):
        """Order points in clockwise order starting from top-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # Top-left
        rect[2] = pts[np.argmax(s)]      # Bottom-right
        rect[1] = pts[np.argmin(diff)]   # Top-right
        rect[3] = pts[np.argmax(diff)]   # Bottom-left
        
        return rect
