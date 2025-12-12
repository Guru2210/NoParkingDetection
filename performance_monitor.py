"""
Performance Monitor
Track system performance metrics including FPS, GPU/CPU usage, and memory
"""

import time
import logging
import psutil
import threading
from collections import deque
from typing import Dict

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, window_size=30, log_interval=10):
        """
        Initialize performance monitor
        
        Args:
            window_size: Number of samples for moving average
            log_interval: Seconds between performance logs
        """
        self.window_size = window_size
        self.log_interval = log_interval
        
        # FPS tracking
        self.frame_times = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        
        # Inference time tracking
        self.inference_times = deque(maxlen=window_size)
        
        # Counters
        self.frame_count = 0
        self.detection_count = 0
        self.violation_count = 0
        
        # System metrics
        self.cpu_percent = 0
        self.memory_percent = 0
        self.gpu_percent = 0
        self.gpu_memory = 0
        
        # Monitoring thread
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_log_time = time.time()
        
        # GPU availability
        self.has_gpu = self._check_gpu()
    
    def _check_gpu(self):
        """Check if GPU monitoring is available"""
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except:
            logger.info("GPU monitoring not available (pynvml not installed)")
            return False
    
    def start(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Log performance if interval elapsed
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    self._log_performance()
                    self.last_log_time = current_time
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    def _update_system_metrics(self):
        """Update CPU, memory, and GPU metrics"""
        try:
            # CPU and memory
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            self.memory_percent = psutil.virtual_memory().percent
            
            # GPU metrics
            if self.has_gpu:
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_percent = utilization.gpu
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory = mem_info.used / mem_info.total * 100
                    
                except Exception as e:
                    logger.debug(f"GPU metrics error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def record_frame(self):
        """Record frame processing"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
        self.frame_count += 1
    
    def record_inference(self, inference_time):
        """
        Record inference time
        
        Args:
            inference_time: Time in seconds
        """
        self.inference_times.append(inference_time)
    
    def record_detection(self):
        """Record detection event"""
        self.detection_count += 1
    
    def record_violation(self):
        """Record violation event"""
        self.violation_count += 1
    
    def get_fps(self):
        """Calculate current FPS"""
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0.0
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds"""
        if len(self.inference_times) == 0:
            return 0.0
        
        return (sum(self.inference_times) / len(self.inference_times)) * 1000
    
    def get_metrics(self) -> Dict:
        """
        Get all performance metrics
        
        Returns:
            Dict with performance metrics
        """
        return {
            'fps': round(self.get_fps(), 2),
            'avg_inference_ms': round(self.get_avg_inference_time(), 2),
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'violation_count': self.violation_count,
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_percent': round(self.memory_percent, 1),
            'gpu_percent': round(self.gpu_percent, 1),
            'gpu_memory_percent': round(self.gpu_memory, 1)
        }
    
    def _log_performance(self):
        """Log performance metrics"""
        metrics = self.get_metrics()
        
        logger.info(
            f"Performance: "
            f"FPS={metrics['fps']:.1f} | "
            f"Inference={metrics['avg_inference_ms']:.1f}ms | "
            f"CPU={metrics['cpu_percent']:.1f}% | "
            f"Memory={metrics['memory_percent']:.1f}% | "
            f"GPU={metrics['gpu_percent']:.1f}% | "
            f"Frames={metrics['frame_count']} | "
            f"Violations={metrics['violation_count']}"
        )
    
    def reset_counters(self):
        """Reset frame and detection counters"""
        self.frame_count = 0
        self.detection_count = 0
        self.violation_count = 0
        logger.info("Performance counters reset")
    
    def get_summary(self):
        """Get performance summary as formatted string"""
        metrics = self.get_metrics()
        
        summary = f"""
Performance Summary:
-------------------
FPS: {metrics['fps']:.2f}
Avg Inference Time: {metrics['avg_inference_ms']:.2f} ms
Total Frames: {metrics['frame_count']}
Total Detections: {metrics['detection_count']}
Total Violations: {metrics['violation_count']}

System Resources:
-----------------
CPU Usage: {metrics['cpu_percent']:.1f}%
Memory Usage: {metrics['memory_percent']:.1f}%
GPU Usage: {metrics['gpu_percent']:.1f}%
GPU Memory: {metrics['gpu_memory_percent']:.1f}%
"""
        return summary
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        logger.info(self.get_summary())
