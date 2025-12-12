"""
Data Models
Define data structures for parking violations
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List


@dataclass
class VehicleDetection:
    """Vehicle detection result"""
    vehicle_id: int
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LicensePlateDetection:
    """License plate detection result"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    text: str = ""
    ocr_confidence: float = 0.0
    valid: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ParkingViolation:
    """Parking violation record"""
    violation_id: str
    vehicle_id: int
    timestamp: datetime
    parking_duration: float  # seconds
    vehicle_bbox: List[int]
    vehicle_class: str
    license_plate: str = ""
    plate_confidence: float = 0.0
    location: str = "Unknown"
    zone: str = "Default"
    status: str = "pending"  # pending, confirmed, resolved
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    vehicle_image_url: Optional[str] = None
    plate_image_url: Optional[str] = None
    notes: str = ""
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class VehicleTrack:
    """Vehicle tracking information"""
    vehicle_id: int
    centroid: tuple  # (x, y)
    start_time: float
    last_seen: float
    bbox_history: List[List[int]] = field(default_factory=list)
    is_parked: bool = False
    parking_start_time: Optional[float] = None
    violation_recorded: bool = False
    
    def get_parking_duration(self):
        """Get current parking duration in seconds"""
        if self.parking_start_time:
            return time.time() - self.parking_start_time
        return 0.0
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'vehicle_id': self.vehicle_id,
            'centroid': self.centroid,
            'start_time': self.start_time,
            'last_seen': self.last_seen,
            'is_parked': self.is_parked,
            'parking_duration': self.get_parking_duration(),
            'violation_recorded': self.violation_recorded
        }


@dataclass
class SystemConfig:
    """System configuration"""
    # Camera
    camera_source: str = "video"
    camera_path: str = ""
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    
    # Detection
    vehicle_model: str = "yolov8n.pt"
    plate_model: str = "NumberPlateDetection.pt"
    use_tensorrt: bool = True
    vehicle_confidence: float = 0.5
    plate_confidence: float = 0.6
    
    # Tracking
    max_disappeared: int = 5
    position_threshold: int = 20
    time_threshold: int = 10
    
    # OCR
    ocr_enabled: bool = True
    ocr_gpu: bool = True
    ocr_languages: List[str] = field(default_factory=lambda: ['en'])
    ocr_confidence: float = 0.7
    
    # Firestore
    firestore_enabled: bool = True
    firestore_credentials: str = "firebase-credentials.json"
    firestore_collection: str = "parking_violations"
    
    # Performance
    max_fps: int = 30
    skip_frames: int = 0
    display_window: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/parking_detector.log"
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)


import time
