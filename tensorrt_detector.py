"""
TensorRT Detector
Optimized inference using TensorRT engines with fallback to PyTorch
"""

import os
import logging
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class TensorRTDetector:
    """Unified detector supporting both TensorRT and PyTorch models"""
    
    def __init__(self, model_path, use_tensorrt=True, confidence=0.5):
        """
        Initialize detector
        
        Args:
            model_path: Path to model file (.pt or .engine)
            use_tensorrt: Whether to use TensorRT if available
            confidence: Confidence threshold for detections
        """
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.confidence = confidence
        self.model = None
        self.is_tensorrt = False
        
        self._load_model()
    
    def _load_model(self):
        """Load model with TensorRT or PyTorch"""
        try:
            # Check if TensorRT engine exists
            tensorrt_path = str(Path(self.model_path).with_suffix('.engine'))
            
            if self.use_tensorrt and os.path.exists(tensorrt_path):
                logger.info(f"Loading TensorRT engine: {tensorrt_path}")
                self.model = YOLO(tensorrt_path, task='detect')
                self.is_tensorrt = True
                logger.info("✓ TensorRT engine loaded successfully")
                
            elif self.use_tensorrt and tensorrt_path.endswith('.engine'):
                # User specified .engine file directly
                if os.path.exists(self.model_path):
                    logger.info(f"Loading TensorRT engine: {self.model_path}")
                    self.model = YOLO(self.model_path, task='detect')
                    self.is_tensorrt = True
                    logger.info("✓ TensorRT engine loaded successfully")
                else:
                    raise FileNotFoundError(f"TensorRT engine not found: {self.model_path}")
            
            else:
                # Fallback to PyTorch
                logger.info(f"Loading PyTorch model: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.is_tensorrt = False
                logger.warning("⚠ Using PyTorch model (slower). Consider converting to TensorRT.")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
            # Try fallback to PyTorch if TensorRT failed
            if self.is_tensorrt:
                logger.info("Attempting fallback to PyTorch model...")
                try:
                    self.model = YOLO(self.model_path)
                    self.is_tensorrt = False
                    logger.info("✓ Fallback to PyTorch successful")
                except Exception as e2:
                    logger.error(f"Fallback failed: {e2}")
                    raise
            else:
                raise
    
    def detect(self, frame, classes=None):
        """
        Run detection on frame
        
        Args:
            frame: Input image (numpy array)
            classes: List of class IDs to detect (None = all classes)
            
        Returns:
            List of detections, each as dict with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence,
                classes=classes,
                verbose=False
            )
            
            detections = []
            
            # Parse results
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(self, frames, classes=None):
        """
        Run detection on batch of frames
        
        Args:
            frames: List of images
            classes: List of class IDs to detect
            
        Returns:
            List of detection lists (one per frame)
        """
        try:
            results = self.model(
                frames,
                conf=self.confidence,
                classes=classes,
                verbose=False
            )
            
            batch_detections = []
            
            for result in results:
                frame_detections = []
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    
                    frame_detections.append(detection)
                
                batch_detections.append(frame_detections)
            
            return batch_detections
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [[] for _ in frames]
    
    def get_model_info(self):
        """Get information about loaded model"""
        return {
            'model_path': self.model_path,
            'is_tensorrt': self.is_tensorrt,
            'confidence': self.confidence,
            'model_loaded': self.model is not None
        }


class VehicleDetector(TensorRTDetector):
    """Specialized detector for vehicles"""
    
    def __init__(self, model_path, use_tensorrt=True, confidence=0.5, 
                 vehicle_classes=[2, 3, 5, 7]):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to model
            use_tensorrt: Use TensorRT if available
            confidence: Detection confidence threshold
            vehicle_classes: COCO class IDs for vehicles
                            [2=car, 3=motorcycle, 5=bus, 7=truck]
        """
        super().__init__(model_path, use_tensorrt, confidence)
        self.vehicle_classes = vehicle_classes
    
    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        return self.detect(frame, classes=self.vehicle_classes)


class LicensePlateDetector(TensorRTDetector):
    """Specialized detector for license plates"""
    
    def __init__(self, model_path, use_tensorrt=True, confidence=0.6):
        """
        Initialize license plate detector
        
        Args:
            model_path: Path to model
            use_tensorrt: Use TensorRT if available
            confidence: Detection confidence threshold
        """
        super().__init__(model_path, use_tensorrt, confidence)
    
    def detect_plates(self, frame):
        """Detect license plates in frame"""
        return self.detect(frame, classes=None)
    
    def detect_plate_in_vehicle(self, frame, vehicle_bbox):
        """
        Detect license plate within vehicle bounding box
        
        Args:
            frame: Full frame
            vehicle_bbox: [x1, y1, x2, y2] of vehicle
            
        Returns:
            List of plate detections with adjusted coordinates
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # Crop vehicle region
        vehicle_crop = frame[y1:y2, x1:x2]
        
        if vehicle_crop.size == 0:
            return []
        
        # Detect plates in crop
        plates = self.detect_plates(vehicle_crop)
        
        # Adjust coordinates to full frame
        for plate in plates:
            px1, py1, px2, py2 = plate['bbox']
            plate['bbox'] = [px1 + x1, py1 + y1, px2 + x1, py2 + y1]
            plate['vehicle_bbox'] = vehicle_bbox
        
        return plates
