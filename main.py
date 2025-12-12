"""
Main Detection System
Production-grade no parking detection with TensorRT, OCR, and Firestore
"""

import cv2
import time
import yaml
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from scipy.spatial import distance as dist

# Import custom modules
from centroid_tracker import CentroidTracker
from tensorrt_detector import VehicleDetector, LicensePlateDetector
from ocr_engine import OCREngine
from firestore_manager import FirestoreManager, LocalViolationStorage
from camera_manager import CameraManager
from performance_monitor import PerformanceMonitor
from models import ParkingViolation

# Setup logging
def setup_logging(log_level='INFO', log_file=None):
    """Configure logging"""
    import colorlog
    
    # Create logs directory
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    
    # File handler
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )

logger = logging.getLogger(__name__)


class ParkingDetectionSystem:
    """Main parking detection system"""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize detection system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Setup logging
        setup_logging(
            log_level=self.config['logging']['level'],
            log_file=self.config['logging'].get('log_file')
        )
        
        logger.info("=" * 60)
        logger.info("Initializing No Parking Detection System")
        logger.info("=" * 60)
        
        # Initialize components
        self.camera = None
        self.vehicle_detector = None
        self.plate_detector = None
        self.ocr_engine = None
        self.firestore = None
        self.local_storage = None
        self.tracker = None
        self.performance = None
        
        # Tracking state
        self.vehicle_positions = {}
        self.vehicle_saved = set()
        
        self._initialize_components()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize camera
            camera_config = self.config['camera']
            self.camera = CameraManager(
                source_type=camera_config['source'],
                video_path=camera_config.get('video_path', ''),
                rtsp_url=camera_config.get('rtsp_url', ''),
                usb_device=camera_config.get('usb_device', 0),
                csi_pipeline=camera_config.get('csi_pipeline', ''),
                width=camera_config.get('width', 1280),
                height=camera_config.get('height', 720),
                fps=camera_config.get('fps', 30),
                reconnect_attempts=camera_config.get('reconnect_attempts', 5),
                reconnect_delay=camera_config.get('reconnect_delay', 5)
            )
            
            # Initialize vehicle detector
            detection_config = self.config['detection']
            model_path = (detection_config['vehicle_model_tensorrt'] 
                         if detection_config['use_tensorrt'] 
                         else detection_config['vehicle_model'])
            
            self.vehicle_detector = VehicleDetector(
                model_path=model_path,
                use_tensorrt=detection_config['use_tensorrt'],
                confidence=detection_config['vehicle_confidence'],
                vehicle_classes=detection_config['vehicle_classes']
            )
            
            # Initialize license plate detector
            plate_model_path = (detection_config['plate_model_tensorrt']
                               if detection_config['use_tensorrt']
                               else detection_config['plate_model'])
            
            self.plate_detector = LicensePlateDetector(
                model_path=plate_model_path,
                use_tensorrt=detection_config['use_tensorrt'],
                confidence=detection_config['plate_confidence']
            )
            
            # Initialize OCR
            if self.config['ocr']['enabled']:
                self.ocr_engine = OCREngine(
                    languages=self.config['ocr']['languages'],
                    gpu=self.config['ocr']['gpu'],
                    confidence_threshold=self.config['ocr']['confidence_threshold']
                )
            
            # Initialize Firestore
            if self.config['firestore']['enabled']:
                self.firestore = FirestoreManager(
                    credentials_path=self.config['firestore']['credentials_path'],
                    collection_name=self.config['firestore']['collection_name'],
                    storage_bucket=self.config['firestore'].get('storage_bucket'),
                    offline_queue=self.config['firestore'].get('offline_queue', True)
                )
            
            # Initialize local storage (fallback)
            self.local_storage = LocalViolationStorage(
                storage_dir=self.config['output']['violations_path']
            )
            
            # Initialize tracker
            self.tracker = CentroidTracker(
                maxDisappeared=self.config['detection']['max_disappeared']
            )
            
            # Initialize performance monitor
            self.performance = PerformanceMonitor(
                window_size=30,
                log_interval=10
            )
            
            logger.info("✓ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of vehicle detections
        """
        start_time = time.time()
        
        # Run vehicle detection
        detections = self.vehicle_detector.detect_vehicles(frame)
        
        # Record inference time
        inference_time = time.time() - start_time
        self.performance.record_inference(inference_time)
        
        return detections
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Args:
            frame: Input frame
        """
        # Detect vehicles
        vehicle_detections = self.detect_vehicles(frame)
        
        # Extract bounding boxes for tracking
        rects = [det['bbox'] for det in vehicle_detections]
        
        # Update tracker
        objects = self.tracker.update(rects)
        
        # Current time
        current_time = time.time()
        
        # Process each tracked vehicle
        for objectID, centroid in objects.items():
            # Initialize tracking if new
            if objectID not in self.vehicle_positions:
                self.vehicle_positions[objectID] = {
                    'centroid': centroid,
                    'start_time': current_time,
                    'bbox': None
                }
            
            # Get previous position
            prev_centroid = self.vehicle_positions[objectID]['centroid']
            
            # Calculate movement
            movement = dist.euclidean(centroid, prev_centroid)
            
            # Check if vehicle is stationary
            position_threshold = self.config['detection']['position_threshold']
            time_threshold = self.config['detection']['time_threshold']
            
            if movement < position_threshold:
                # Vehicle is stationary
                elapsed_time = current_time - self.vehicle_positions[objectID]['start_time']
                
                # Check if exceeded time threshold
                if elapsed_time > time_threshold and objectID not in self.vehicle_saved:
                    # Find corresponding detection
                    det_idx = objectID % len(vehicle_detections) if vehicle_detections else 0
                    
                    if det_idx < len(vehicle_detections):
                        detection = vehicle_detections[det_idx]
                        bbox = detection['bbox']
                        
                        # Record violation
                        self._record_violation(
                            frame=frame,
                            vehicle_id=objectID,
                            bbox=bbox,
                            parking_duration=elapsed_time,
                            vehicle_class=detection['class_name']
                        )
                        
                        # Mark as saved
                        self.vehicle_saved.add(objectID)
                        self.performance.record_violation()
                        
                        # Draw violation indicator
                        cv2.putText(
                            frame,
                            "UNAUTHORIZED PARKING!",
                            (centroid[0] - 100, centroid[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )
            else:
                # Vehicle moved, reset timer
                self.vehicle_positions[objectID]['centroid'] = centroid
                self.vehicle_positions[objectID]['start_time'] = current_time
            
            # Draw tracking visualization
            self._draw_tracking(frame, objectID, centroid, rects)
        
        # Record frame processing
        self.performance.record_frame()
        
        return frame
    
    def _record_violation(self, frame, vehicle_id, bbox, parking_duration, vehicle_class):
        """
        Record parking violation
        
        Args:
            frame: Current frame
            vehicle_id: Tracked vehicle ID
            bbox: Vehicle bounding box
            parking_duration: Duration in seconds
            vehicle_class: Vehicle class name
        """
        try:
            logger.warning(f"⚠ Violation detected! Vehicle ID: {vehicle_id}, Duration: {parking_duration:.1f}s")
            
            # Crop vehicle image
            x1, y1, x2, y2 = bbox
            vehicle_crop = frame[y1:y2, x1:x2]
            
            # Detect license plate
            plate_text = ""
            plate_confidence = 0.0
            plate_crop = None
            
            plates = self.plate_detector.detect_plate_in_vehicle(frame, bbox)
            
            if plates and self.ocr_engine:
                # Get best plate
                best_plate = max(plates, key=lambda p: p['confidence'])
                px1, py1, px2, py2 = best_plate['bbox']
                plate_crop = frame[py1:py2, px1:px2]
                
                # Run OCR
                ocr_result = self.ocr_engine.read_license_plate(plate_crop)
                plate_text = ocr_result['text']
                plate_confidence = ocr_result['confidence']
                
                logger.info(f"License Plate: {plate_text} (confidence: {plate_confidence:.2f})")
            
            # Create violation record
            violation_id = f"VIO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{vehicle_id}"
            
            violation = ParkingViolation(
                violation_id=violation_id,
                vehicle_id=vehicle_id,
                timestamp=datetime.now(),
                parking_duration=parking_duration,
                vehicle_bbox=bbox,
                vehicle_class=vehicle_class,
                license_plate=plate_text,
                plate_confidence=plate_confidence,
                location=self.config.get('location', 'Unknown'),
                zone=self.config.get('zone', 'Default')
            )
            
            # Save to Firestore
            if self.firestore and self.firestore.is_connected:
                self.firestore.save_violation(
                    violation.to_dict(),
                    vehicle_image=vehicle_crop,
                    plate_image=plate_crop
                )
            
            # Save locally (always)
            self.local_storage.save_violation(
                violation.to_dict(),
                vehicle_image=vehicle_crop,
                plate_image=plate_crop
            )
            
        except Exception as e:
            logger.error(f"Failed to record violation: {e}")
    
    def _draw_tracking(self, frame, objectID, centroid, rects):
        """Draw tracking visualization"""
        # Draw centroid
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        # Draw ID
        cv2.putText(
            frame,
            f"ID {objectID}",
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Draw bounding box if available
        if rects and objectID < len(rects):
            x1, y1, x2, y2 = rects[objectID % len(rects)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def run(self):
        """Run main detection loop"""
        try:
            logger.info("Starting detection system...")
            
            # Start camera
            self.camera.start()
            
            # Start performance monitoring
            self.performance.start()
            
            # Video writer (if enabled)
            video_writer = None
            if self.config['output']['save_video']:
                output_path = self.config['output']['video_output_path']
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                width, height = self.camera.get_resolution()
                fps = self.camera.get_fps()
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info(f"Saving output video to: {output_path}")
            
            logger.info("✓ Detection system running. Press 'q' to quit.")
            
            # Main loop
            while True:
                # Read frame
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Save to video
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Display (if enabled)
                if self.config['output']['display_window']:
                    cv2.imshow("No Parking Detection", processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit signal received")
                        break
                
                # FPS limiting
                max_fps = self.config['performance']['max_fps']
                if max_fps > 0:
                    time.sleep(1.0 / max_fps)
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        
        if self.camera:
            self.camera.stop()
        
        if self.performance:
            self.performance.stop()
        
        if video_writer:
            video_writer.release()
        
        cv2.destroyAllWindows()
        
        logger.info("✓ Cleanup complete")
        logger.info(self.performance.get_summary())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='No Parking Detection System'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run system
        system = ParkingDetectionSystem(config_path=args.config)
        system.run()
        
    except Exception as e:
        logger.error(f"System failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
