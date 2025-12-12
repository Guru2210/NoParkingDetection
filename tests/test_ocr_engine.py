"""
Unit tests for OCR Engine
"""

import pytest
import cv2
import numpy as np
from ocr_engine import OCREngine, PlatePreprocessor


class TestOCREngine:
    """Test OCR engine functionality"""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCR engine instance"""
        return OCREngine(languages=['en'], gpu=False, confidence_threshold=0.5)
    
    def test_initialization(self, ocr_engine):
        """Test OCR engine initialization"""
        assert ocr_engine is not None
        assert ocr_engine.reader is not None
        assert ocr_engine.confidence_threshold == 0.5
    
    def test_clean_text(self, ocr_engine):
        """Test text cleaning"""
        assert ocr_engine._clean_text("ABC 123") == "ABC123"
        assert ocr_engine._clean_text("abc-123") == "ABC123"
        assert ocr_engine._clean_text("  XYZ 789  ") == "XYZ789"
    
    def test_validate_license_plate(self, ocr_engine):
        """Test license plate validation"""
        assert ocr_engine._validate_license_plate("ABC123") == True
        assert ocr_engine._validate_license_plate("XYZ789DEF") == True
        assert ocr_engine._validate_license_plate("123") == False  # Too short
        assert ocr_engine._validate_license_plate("ABCDEFGHIJKLM") == False  # Too long
        assert ocr_engine._validate_license_plate("ABCDEF") == False  # No numbers
        assert ocr_engine._validate_license_plate("123456") == False  # No letters
    
    def test_preprocess_image(self, ocr_engine):
        """Test image preprocessing"""
        # Create test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Preprocess
        processed = ocr_engine._preprocess_image(test_image)
        
        # Check output
        assert processed is not None
        assert len(processed.shape) == 2  # Should be grayscale
    
    def test_read_text_empty_image(self, ocr_engine):
        """Test OCR on empty image"""
        empty_image = np.zeros((100, 200, 3), dtype=np.uint8)
        results = ocr_engine.read_text(empty_image)
        
        # Should return empty list or handle gracefully
        assert isinstance(results, list)


class TestPlatePreprocessor:
    """Test plate preprocessor"""
    
    def test_enhance_plate(self):
        """Test plate enhancement"""
        # Create test image
        test_image = np.random.randint(0, 255, (64, 320, 3), dtype=np.uint8)
        
        # Enhance
        enhanced = PlatePreprocessor.enhance_plate(test_image)
        
        # Check output
        assert enhanced is not None
        assert enhanced.shape[0] > 0
        assert enhanced.shape[1] > 0
    
    def test_order_points(self):
        """Test point ordering"""
        # Create test points (random quadrilateral)
        pts = np.array([
            [100, 100],
            [200, 100],
            [200, 200],
            [100, 200]
        ], dtype=np.float32)
        
        # Shuffle
        np.random.shuffle(pts)
        
        # Order
        ordered = PlatePreprocessor._order_points(pts)
        
        # Check that we got 4 points back
        assert ordered.shape == (4, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
