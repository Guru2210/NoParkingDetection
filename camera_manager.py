"""
Camera Manager
Unified interface for multiple camera sources with auto-reconnection
"""

import cv2
import logging
import threading
import time
from queue import Queue
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CameraManager:
    """Manage camera input with support for multiple sources"""
    
    def __init__(self, source_type='video', **kwargs):
        """
        Initialize camera manager
        
        Args:
            source_type: 'video', 'rtsp', 'usb', or 'csi'
            **kwargs: Source-specific parameters
        """
        self.source_type = source_type
        self.kwargs = kwargs
        self.cap = None
        self.is_running = False
        self.reconnect_attempts = kwargs.get('reconnect_attempts', 5)
        self.reconnect_delay = kwargs.get('reconnect_delay', 5)
        
        # Frame buffering
        self.frame_queue = Queue(maxsize=30)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Threading
        self.capture_thread = None
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera based on source type"""
        try:
            if self.source_type == 'video':
                self._init_video()
            elif self.source_type == 'rtsp':
                self._init_rtsp()
            elif self.source_type == 'usb':
                self._init_usb()
            elif self.source_type == 'csi':
                self._init_csi()
            else:
                raise ValueError(f"Unknown source type: {self.source_type}")
            
            if self.cap and self.cap.isOpened():
                logger.info(f"✓ Camera initialized: {self.source_type}")
                self._log_camera_info()
            else:
                raise RuntimeError("Failed to open camera")
                
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
    
    def _init_video(self):
        """Initialize video file source"""
        video_path = self.kwargs.get('video_path', '')
        if not video_path:
            raise ValueError("video_path required for video source")
        
        self.cap = cv2.VideoCapture(video_path)
        logger.info(f"Loading video: {video_path}")
    
    def _init_rtsp(self):
        """Initialize RTSP stream"""
        rtsp_url = self.kwargs.get('rtsp_url', '')
        if not rtsp_url:
            raise ValueError("rtsp_url required for RTSP source")
        
        # RTSP optimization flags
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        logger.info(f"Connecting to RTSP: {rtsp_url}")
    
    def _init_usb(self):
        """Initialize USB camera"""
        device_id = self.kwargs.get('usb_device', 0)
        self.cap = cv2.VideoCapture(device_id)
        
        # Set resolution if specified
        width = self.kwargs.get('width', 1280)
        height = self.kwargs.get('height', 720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Set FPS if specified
        fps = self.kwargs.get('fps', 30)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        logger.info(f"Opening USB camera: {device_id}")
    
    def _init_csi(self):
        """Initialize CSI camera (Jetson Nano)"""
        # Default GStreamer pipeline for CSI camera on Jetson
        pipeline = self.kwargs.get('csi_pipeline', '')
        
        if not pipeline:
            width = self.kwargs.get('width', 1280)
            height = self.kwargs.get('height', 720)
            fps = self.kwargs.get('fps', 30)
            
            pipeline = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                f"format=NV12, framerate={fps}/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, width={width}, height={height}, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! appsink"
            )
        
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        logger.info("Opening CSI camera with GStreamer")
    
    def _log_camera_info(self):
        """Log camera properties"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            logger.info(f"Camera properties: {width}x{height} @ {fps} FPS")
    
    def start(self):
        """Start camera capture in separate thread"""
        if self.is_running:
            logger.warning("Camera already running")
            return
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Camera capture started")
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        consecutive_failures = 0
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.warning("Camera disconnected, attempting reconnection...")
                    self._reconnect()
                    continue
                
                ret, frame = self.cap.read()
                
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame
                    
                    # Update queue (drop old frames if full)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    
                    self.frame_queue.put(frame)
                    consecutive_failures = 0
                    
                else:
                    consecutive_failures += 1
                    
                    if consecutive_failures > 30:
                        logger.error("Too many consecutive frame read failures")
                        self._reconnect()
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(1)
    
    def _reconnect(self):
        """Attempt to reconnect camera"""
        for attempt in range(self.reconnect_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{self.reconnect_attempts}")
            
            try:
                if self.cap:
                    self.cap.release()
                
                time.sleep(self.reconnect_delay)
                
                self._initialize_camera()
                
                if self.cap and self.cap.isOpened():
                    logger.info("✓ Reconnection successful")
                    return True
                    
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
        
        logger.error("All reconnection attempts failed")
        self.is_running = False
        return False
    
    def read(self) -> Tuple[bool, Optional[any]]:
        """
        Read current frame
        
        Returns:
            (success, frame) tuple
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            else:
                return False, None
    
    def get_frame(self, timeout=1.0):
        """
        Get frame from queue (blocking)
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame or None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None
    
    def stop(self):
        """Stop camera capture"""
        logger.info("Stopping camera capture...")
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Camera stopped")
    
    def is_opened(self):
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self):
        """Get camera FPS"""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0
    
    def get_resolution(self):
        """Get camera resolution"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
